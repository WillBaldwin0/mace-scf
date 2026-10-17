"""Four-function electronic densities and graph_longrange electrostatics.

The basis is (g, M_1,-1, M_1,0, M_1,1), with coefficients (charge, dy, dz, dx).
Charges are in units of e; lengths in Angstrom and energies in eV. The omitted
quadrupoles and S->g approximation are choices of density map, not of A.
"""

import math

import torch
from torch import nn
from graph_longrange.energy import GTOElectrostaticEnergy


class ShellCoupling(nn.Module):
    def __init__(self, n_s, n_p, coupling_mode="paired", C=None, trainable=False):
        super().__init__()
        if min(n_s, n_p) < 0 or n_s + n_p == 0:
            raise ValueError("Require a nonempty s/p basis")
        if trainable:
            raise NotImplementedError(
                "Learned A is reserved for a later implementation"
            )
        self.n_s, self.n_p = n_s, n_p
        if coupling_mode == "paired":
            C = torch.eye(n_s, n_p)
        elif coupling_mode == "zero":
            C = torch.zeros(n_s, n_p)
        elif coupling_mode == "specified":
            if C is None:
                raise ValueError("specified shell coupling requires C")
            C = torch.as_tensor(C, dtype=torch.get_default_dtype())
        else:
            raise ValueError("coupling_mode must be paired, zero or specified")
        if C.shape != (n_s, n_p) or not bool(torch.isfinite(C).all()):
            raise ValueError("C must be a finite [n_s,n_p] tensor")
        if C.numel() and float(torch.linalg.svdvals(C).max()) > 1 + 1e-7:
            raise ValueError("The s-p coupling C must be a contraction")
        self.register_buffer("C", C.clone())
        self.register_buffer("eye_s", torch.eye(n_s))
        self.register_buffer("eye_p", torch.eye(n_p))

    def forward(self, gamma):
        n, s, p = gamma.shape[0], self.n_s, self.n_p
        if gamma.shape[1:] != (s + 3 * p, s + 3 * p):
            raise ValueError("Density matrix has the wrong electronic basis")
        ss = gamma[:, :s, :s].diagonal(dim1=-2, dim2=-1).sum(-1)
        sp = torch.einsum("nspa,sp->na", gamma[:, :s, s:].reshape(n, s, p, 3), self.C)
        ps = torch.einsum("npas,sp->na", gamma[:, s:, :s].reshape(n, p, 3, s), self.C)
        pp = torch.einsum(
            "nuavb,uv->nab", gamma[:, s:, s:].reshape(n, p, 3, p, 3), self.eye_p
        )
        return torch.cat(
            (
                torch.cat((ss[:, None, None], sp[:, None, :]), -1),
                torch.cat((ps[:, :, None], pp), -1),
            ),
            -2,
        )

    def adjoint(self, matrix):
        n, s, p = matrix.shape[0], self.n_s, self.n_p
        ss = matrix[:, :1, :1] * self.eye_s
        sp = torch.einsum("na,sp->nspa", matrix[:, 0, 1:], self.C).reshape(n, s, 3 * p)
        ps = torch.einsum("na,sp->npas", matrix[:, 1:, 0], self.C).reshape(n, 3 * p, s)
        pp = torch.einsum("nab,uv->nuavb", matrix[:, 1:, 1:], self.eye_p).reshape(
            n, 3 * p, 3 * p
        )
        return torch.cat((torch.cat((ss, sp), -1), torch.cat((ps, pp), -1)), -2)


class DensityBasis(nn.Module):
    def __init__(self, sigma=1.0, pp_scalar="g", keep_quadrupoles=False):
        super().__init__()
        if pp_scalar not in {"g", "S"}:
            raise ValueError("pp_scalar must be g or S")
        if pp_scalar != "g" or keep_quadrupoles:
            raise NotImplementedError(
                "Only the four-function basis (pp_scalar='g', "
                "keep_quadrupoles=False) is implemented"
            )
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be positive and finite")
        self.sigma = float(sigma)
        self.pp_scalar, self.keep_quadrupoles = pp_scalar, keep_quadrupoles
        self.num_channels = 4

    def moments(self, coefficients):
        return coefficients[:, 0], coefficients[:, [3, 1, 2]]


class ProductDensityMap(nn.Module):
    def __init__(self, basis):
        super().__init__()
        self.basis = basis
        T = torch.zeros(4, 4, 4)  # [channel, primitive row, primitive column]
        T[0] = torch.eye(4)
        for k, axis in enumerate((2, 3, 1), start=1):
            T[k, 0, axis] = T[k, axis, 0] = basis.sigma
        self.register_buffer("T", T)

    def forward(self, D):
        return torch.einsum("nab,kab->nk", D, self.T)

    def adjoint(self, u):
        return torch.einsum("nk,kab->nab", u, self.T)


class ElectronicDensityMap(nn.Module):
    def __init__(self, shell_coupling, product_map):
        super().__init__()
        self.shell_coupling, self.product_map = shell_coupling, product_map

    def forward(self, gamma):
        return self.product_map(self.shell_coupling(gamma))

    def adjoint(self, u):
        return self.shell_coupling.adjoint(self.product_map.adjoint(u))


class NuclearSourceModel(nn.Module):
    def __init__(
        self, effective_charges, profile="shared_g", charge_mode="per_species"
    ):
        super().__init__()
        if profile != "shared_g":
            raise NotImplementedError("Separate nuclear Gaussians are not implemented")
        if charge_mode not in {"per_species", "from_data"}:
            raise ValueError("charge_mode must be per_species or from_data")
        values = torch.as_tensor(effective_charges, dtype=torch.get_default_dtype())
        if (
            values.ndim != 1
            or not bool(torch.isfinite(values).all())
            or bool((values < 0).any())
        ):
            raise ValueError(
                "Effective nuclear charges must be finite nonnegative species values"
            )
        self.register_buffer("effective_charges", values)
        self.charge_mode = charge_mode

    def forward(self, q, species, effective_nuclear_charges=None):
        if self.charge_mode == "from_data":
            if effective_nuclear_charges is None:
                raise ValueError(
                    "effective_nuclear_charges are required in from_data mode"
                )
            Z = effective_nuclear_charges.to(q)
        else:
            if effective_nuclear_charges is not None:
                raise ValueError(
                    "Use from_data mode to supply nuclear charge overrides"
                )
            Z = self.effective_charges[species]
        if (
            Z.shape != q[:, 0].shape
            or not bool(torch.isfinite(Z).all())
            or bool((Z < 0).any())
        ):
            raise ValueError("Invalid per-atom effective nuclear charges")
        nuclear = torch.cat((Z[:, None], torch.zeros_like(q[:, 1:])), dim=-1)
        return nuclear - q, nuclear, Z


class DensityElectrostatics(nn.Module):
    """Map electronic densities to multipoles and evaluate their GLR energy."""

    def __init__(
        self,
        n_s,
        n_p,
        effective_charges,
        sigma=1.0,
        pp_scalar="g",
        keep_quadrupoles=False,
        coupling_mode="paired",
        C=None,
        nuclear_profile="shared_g",
        nuclear_charge_mode="per_species",
        self_policy="full",
        pbc_handling="auto",
        *,
        kspace_cutoff,
    ):
        super().__init__()
        if self_policy not in {"full", "subtract_atomic"}:
            raise ValueError("self_policy must be full or subtract_atomic")
        self.basis = DensityBasis(sigma, pp_scalar, keep_quadrupoles)
        self.density_map = ElectronicDensityMap(
            ShellCoupling(n_s, n_p, coupling_mode, C), ProductDensityMap(self.basis)
        )
        self.nuclei = NuclearSourceModel(
            effective_charges, nuclear_profile, nuclear_charge_mode
        )
        self.coulomb_energy = GTOElectrostaticEnergy(
            density_max_l=1,
            density_smearing_width=sigma,
            kspace_cutoff=kspace_cutoff,
            include_self_interaction=self_policy == "full",
            pbc_handling=pbc_handling,
        )

    def multipoles(self, gamma, gamma_spin, species, effective_nuclear_charges=None):
        """Return electronic spin multipoles and signed nuclear-plus-electron multipoles."""
        if gamma_spin.shape != gamma.shape:
            raise ValueError("gamma_spin must have the same shape as gamma")
        total = self.density_map(gamma)
        difference = self.density_map(gamma_spin)
        net_charge, _, _ = self.nuclei(total, species, effective_nuclear_charges)
        return {
            "electron_alpha": 0.5 * (total + difference),
            "electron_beta": 0.5 * (total - difference),
            "net_charge": net_charge,
        }

    def electrostatic_energy(self, net_charge, geometry, external_field=None):
        """Evaluate energy with prepared reciprocal geometry and field = +grad(V)."""
        num_graphs = geometry["volume"].numel()
        coulomb = self.coulomb_energy(
            source_feats=net_charge,
            node_positions=geometry["positions"],
            batch=geometry["batch"],
            volume=geometry["volume"],
            pbc=geometry["pbc"],
            k_vectors=geometry["k_vectors"],
            k_norm2=geometry["k_norm2"],
            k_vector_batch=geometry["k_vector_batch"],
            k0_mask=geometry["k0_mask"],
        ).reshape(num_graphs)
        field_energy = net_charge.new_zeros(num_graphs)
        if external_field is not None:
            field = external_field.to(net_charge).reshape(num_graphs, 3)
            charge, intrinsic = self.basis.moments(net_charge)
            atomic_dipole = charge[:, None] * geometry["positions"] + intrinsic
            node_energy = (atomic_dipole * field[geometry["batch"]]).sum(-1)
            field_energy = field_energy.index_add(0, geometry["batch"], node_energy)
        return {
            "electrostatic_energy": coulomb + field_energy,
            "electrostatic_contributions": {"coulomb": coulomb, "field": field_energy},
        }

    def effective_potential(self, *args, **kwargs):
        raise NotImplementedError("Effective potential is not implemented")

    def forward(
        self,
        gamma,
        gamma_spin,
        geometry,
        species,
        external_field=None,
        effective_nuclear_charges=None,
        return_density=False,
    ):
        multipoles = self.multipoles(
            gamma, gamma_spin, species, effective_nuclear_charges
        )
        q = multipoles["electron_alpha"] + multipoles["electron_beta"]
        spin = multipoles["electron_alpha"] - multipoles["electron_beta"]
        c = multipoles["net_charge"]
        ng = geometry["volume"].numel()
        charge, intrinsic = self.basis.moments(c)
        dipole = q.new_zeros(ng, 3).index_add(
            0, geometry["batch"], charge[:, None] * geometry["positions"] + intrinsic
        )
        result = dict(
            multipoles=multipoles,
            atomic_electron_counts=q[:, 0],
            atomic_charges=charge,
            atomic_dipoles=intrinsic,
            atomic_multipoles=c,
            atomic_spin=spin[:, 0],
            dipole=dipole,
            total_charge=q.new_zeros(ng).index_add(0, geometry["batch"], charge),
            effective_nuclear_charges=c[:, 0] + q[:, 0],
            density_coefficients=c,
        )
        if return_density:
            result.update(
                electron_coefficients=q, spin_coefficients=spin, charge_coefficients=c
            )
        result.update(self.electrostatic_energy(c, geometry, external_field))
        return result
