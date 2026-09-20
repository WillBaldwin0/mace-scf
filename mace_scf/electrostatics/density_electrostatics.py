"""Four-function electronic densities and graph_longrange electrostatics.

The basis is (g, M_1,-1, M_1,0, M_1,1), with coefficients (charge, dy, dz, dx).
Charges are in units of e; lengths in Angstrom and energies in eV. The omitted
quadrupoles and S->g approximation are choices of density map, not of A.
"""

import math

import torch
from torch import nn
from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.features import GTOElectrostaticFeatures
from graph_longrange.gto_utils import gto_basis_kspace_cutoff
from graph_longrange.kspace import compute_k_vectors_flat
from graph_longrange.utils import CUBIC_MADELUNG


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


class ElectrostaticEnergy(nn.Module):
    """Per-graph dispatch supports cell-free molecules and mixed periodic batches.

    'auto' uses realspace for molecules, pbc for 3D and slab for xy-periodic
    cells. Realspace inherits graph_longrange's finite-difference dipole kernel.
    Its explicit projection stencil is aligned with that of the energy.
    """

    def __init__(
        self, basis, self_policy="full", pbc_handling="auto", kspace_cutoff_factor=1.5
    ):
        super().__init__()
        if self_policy not in {"full", "subtract_nuclear", "subtract_atomic"}:
            raise ValueError("Unknown self-interaction policy")
        modes = ("realspace", "pbc", "slab", "molecule_in_box")
        if pbc_handling not in (*modes, "auto", "mixed_periodic"):
            raise ValueError("Unknown boundary handling mode")
        if not math.isfinite(kspace_cutoff_factor) or kspace_cutoff_factor <= 0:
            raise ValueError("kspace_cutoff_factor must be positive")
        self.self_policy, self.pbc_handling = self_policy, pbc_handling
        cutoff = kspace_cutoff_factor * gto_basis_kspace_cutoff([basis.sigma], 1)
        self.register_buffer("kspace_cutoff", torch.tensor(cutoff))
        self.energies = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        for mode in modes:
            self.energies[mode] = GTOElectrostaticEnergy(
                1, basis.sigma, cutoff, include_self_interaction=True, pbc_handling=mode
            )
            self.projections[mode] = GTOElectrostaticFeatures(
                1,
                basis.sigma,
                1,
                [basis.sigma],
                True,
                cutoff,
                integral_normalization="multipoles",
                pbc_handling=mode,
            )
            # graph_longrange currently uses 0.02 for energy, 0.1 for features.
            # Align both the displacements and inverse-offset normalization.
            f = self.projections[mode].realspace_features
            offset = self.energies[mode].realspace_energy.offset
            ratio = offset / f.offset
            for name in ("x", "y", "z"):
                getattr(f, name).mul_(ratio)
            f.l1_factors.div_(ratio)
            f.offset = offset

    def _mode(self, pbc):
        if self.pbc_handling in {"auto", "mixed_periodic"}:
            if not bool(pbc.any()):
                return (
                    "molecule_in_box"
                    if self.pbc_handling == "mixed_periodic"
                    else "realspace"
                )
            if bool(pbc.all()):
                return "pbc"
            if pbc.tolist() == [True, True, False]:
                return "slab"
            raise NotImplementedError(
                "Only open, 3D and xy-slab boundaries are supported"
            )
        if self.pbc_handling in {"realspace", "molecule_in_box"} and bool(pbc.any()):
            raise ValueError("Selected electrostatics mode requires open boundaries")
        if self.pbc_handling == "pbc" and not bool(pbc.all()):
            raise ValueError("pbc requires three periodic axes")
        if self.pbc_handling == "slab" and pbc.tolist() != [True, True, False]:
            raise ValueError("slab requires xy periodicity")
        return self.pbc_handling

    def forward(
        self,
        c,
        nuclear,
        geometry,
        external_field,
        compute_energy=True,
        compute_projections=False,
    ):
        positions, cells, pbc, ptr = (
            geometry[k] for k in ("positions", "cell", "pbc", "ptr")
        )
        full, corrections, fields, projected = [], [], [], []
        for graph, (first, last) in enumerate(zip(ptr[:-1].tolist(), ptr[1:].tolist())):
            x, source = positions[first:last], c[first:last]
            mode = self._mode(pbc[graph])
            cell = cells[graph : graph + 1]
            batch = torch.zeros(last - first, dtype=torch.long, device=c.device)
            if mode == "realspace":
                kv, kn = c.new_empty((0, 3)), c.new_empty(0)
                kb = torch.empty(0, dtype=torch.long, device=c.device)
                k0, volume = c.new_empty(0), c.new_ones(1)
            else:
                volume = torch.linalg.det(cell)
                if not bool((volume > 1e-10).all()):
                    raise ValueError(
                        "Reciprocal electrostatics requires a right-handed, nonsingular cell"
                    )
                reciprocal = 2 * math.pi * torch.linalg.inv(cell.mT)
                kv, kn, kb, k0 = compute_k_vectors_flat(
                    self.kspace_cutoff, cell, reciprocal
                )
            kwargs = dict(
                k_vectors=kv,
                k_norm2=kn,
                k_vector_batch=kb,
                k0_mask=k0,
                source_feats=source,
                node_positions=x,
                batch=batch,
                volume=volume,
                pbc=pbc[graph : graph + 1],
            )
            engine = self.energies[mode]
            full.append(
                engine(**kwargs).reshape(()) if compute_energy else c.new_zeros(())
            )
            correction = source.sum() * 0
            if self.self_policy != "full":
                subtract = (
                    nuclear[first:last]
                    if self.self_policy == "subtract_nuclear"
                    else source
                )
                self_fields = engine.self_interaction_terms(subtract)
                correction = -0.5 * (subtract * self_fields).sum()
            corrections.append(correction)
            dipole = (source[:, :1] * x + source[:, [3, 1, 2]]).sum(0)
            fields.append(-(dipole * external_field[graph]).sum())
            if compute_projections:
                projection_mode = "pbc" if mode == "molecule_in_box" else mode
                w = self.projections[projection_mode](**kwargs)
                if mode == "molecule_in_box":
                    # Explicit adjoint of the library's finite-box energy correction.
                    # Its feature correction currently uses a different exponent
                    # for L=V**(1/3), so it is not exactly the energy derivative.
                    Q = source[:, 0].sum()
                    local_dipoles = source[:, [3, 1, 2]]
                    moment = (source[:, :1] * x + local_dipoles).sum(0)
                    radius2 = x.square().sum(-1)
                    second = (
                        source[:, 0] * radius2 + 2 * (local_dipoles * x).sum(-1)
                    ).sum()
                    const = engine.monopole_dipole_correction.const
                    factor = 2 * math.pi * const / (3 * volume[0])
                    scalar = CUBIC_MADELUNG * const * Q / volume[0] ** 0.3333
                    scalar = scalar + factor * (
                        2 * (x * moment).sum(-1) - second - Q * radius2
                    )
                    dipolar = 2 * factor * (moment - Q * x)
                    w = w + torch.cat((scalar[:, None], dipolar[:, [1, 2, 0]]), -1)
                if self.self_policy == "subtract_atomic":
                    w = w - engine.self_interaction_terms(source)
                # Signed-charge coefficient derivative of -mu.E.
                field = external_field[graph]
                w = w + torch.cat(
                    (
                        -(x * field).sum(-1, keepdim=True),
                        -field[[1, 2, 0]].expand(last - first, 3),
                    ),
                    dim=-1,
                )
                projected.append(w)
                # The library exposes a cache; don't retain this forward's graph.
                self.projections[projection_mode].static_quantities = None
        contributions = dict(
            coulomb_full=torch.stack(full),
            self_correction=torch.stack(corrections),
            field=torch.stack(fields),
        )
        return dict(
            energy=sum(contributions.values()),
            contributions=contributions,
            projections=torch.cat(projected) if compute_projections else None,
        )


class DensityElectrostatics(nn.Module):
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
        kspace_cutoff_factor=1.5,
    ):
        super().__init__()
        self.basis = DensityBasis(sigma, pp_scalar, keep_quadrupoles)
        self.density_map = ElectronicDensityMap(
            ShellCoupling(n_s, n_p, coupling_mode, C), ProductDensityMap(self.basis)
        )
        self.nuclei = NuclearSourceModel(
            effective_charges, nuclear_profile, nuclear_charge_mode
        )
        self.electrostatics = ElectrostaticEnergy(
            self.basis, self_policy, pbc_handling, kspace_cutoff_factor
        )

    def forward(
        self,
        gamma,
        geometry,
        species,
        gamma_spin=None,
        external_field=None,
        effective_nuclear_charges=None,
        compute_energy=True,
        compute_potential=False,
        return_density=False,
    ):
        q = self.density_map(gamma)
        c, nuclear, Z = self.nuclei(q, species, effective_nuclear_charges)
        ng = geometry["ptr"].numel() - 1
        if external_field is None:
            external_field = q.new_zeros(ng, 3)
        else:
            external_field = external_field.to(q).reshape(ng, 3)
        charge, intrinsic = self.basis.moments(c)
        dipole = q.new_zeros(ng, 3).index_add(
            0, geometry["batch"], charge[:, None] * geometry["positions"] + intrinsic
        )
        result = dict(
            atomic_electron_counts=q[:, 0],
            atomic_charges=charge,
            atomic_dipoles=intrinsic,
            atomic_multipoles=c,
            dipole=dipole,
            total_charge=q.new_zeros(ng).index_add(0, geometry["batch"], charge),
            effective_nuclear_charges=Z,
            density_coefficients=c,
        )
        if gamma_spin is not None:
            spin = self.density_map(gamma_spin)
            result["atomic_spin"] = spin[:, 0]
            if return_density:
                result["spin_coefficients"] = spin
        if return_density:
            result.update(electron_coefficients=q, charge_coefficients=c)
        if compute_energy or compute_potential:
            es = self.electrostatics(
                c, nuclear, geometry, external_field, compute_energy, compute_potential
            )
            if compute_energy:
                result.update(
                    electrostatic_energy=es["energy"],
                    electrostatic_contributions=es["contributions"],
                )
            if compute_potential:
                result["effective_potential"] = self.density_map.adjoint(
                    -es["projections"]
                )
        return result
