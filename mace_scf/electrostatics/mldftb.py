"""Non-self-consistent ML-DFTB with a shared s/p electronic basis.

Inputs follow MACE's batched graph dictionary, with graph-level N_alpha,
N_beta and total_charge. Optional elec_temp values use elec_temp_units;
missing temperatures use default_elec_temp (0.25 eV). All energies are eV.
A minimal construction is ``OneShotMLRKS([1, 8], [1., 6.], n_s=1, n_p=1)``;
pass ``model(batch, training=True)`` when fitting energies and forces. Effective
nuclear charges are model parameters chosen with the electron counts, not
necessarily atomic numbers. Use double precision for the electrostatic kernels.
Periodic Hamiltonians are real Gamma-point matrices. Positive elec_temp is
required for force training; zero-width energies and ordinary forces are supported.

The Hamiltonian is evaluated once, without electrostatic self-consistency.
"""

import math

import numpy as np
import torch
from e3nn import o3
from graph_longrange.gto_utils import gto_basis_kspace_cutoff
from graph_longrange.kspace import compute_k_vectors_flat
from mace.modules import (
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)
from .localsources import _LocalSourceModelBase
from .matrix_ops import (
    HamiltonianBuilder,
    OnsiteHamiltonianHead,
    ElectronicState,
    OnsiteExchangeCorrelation,
    MultipoleExchangeCorrelation,
)
from .density_electrostatics import DensityElectrostatics


class _OneShotMLBase(_LocalSourceModelBase):
    def __init__(
        self,
        atomic_numbers,
        effective_nuclear_charges=None,
        n_s=1,
        n_p=1,
        r_max=5.0,
        hidden_irreps="16x0e + 16x1o",
        num_interactions=2,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        correlation=2,
        MLP_irreps="16x0e",
        avg_num_neighbors=8.0,
        atomic_energies=None,
        radial_MLP=None,
        radial_type="bessel",
        interaction_cls=RealAgnosticResidualInteractionBlock,
        interaction_cls_first=RealAgnosticInteractionBlock,
        gate=torch.nn.functional.silu,
        heads=None,
        edge_mode="tensor_product",
        matrix_feature_multiplicity=64,
        matrix_sh_max_l=1,
        dos_grid=None,
        dos_width=0.2,
        higher_order_backend="smooth",
        spectral_chunk_size=None,
        matrix_radial_hidden=32,
        elec_temp_units="kelvin",
        default_elec_temp=0.25,
        energy_kind="internal",
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
        hamiltonian_cutoff=None,
        xc_type="onsite",
        xc_readout="nonlinear",
        xc_channels=16,
        xc_hidden=64,
        xc_norm_epsilon=0.01,
    ):
        super().__init__()
        if energy_kind not in {"internal", "free"}:
            raise ValueError("energy_kind must be internal or free")
        if num_interactions < 1:
            raise ValueError("At least one interaction is required")
        if effective_nuclear_charges is None:
            if nuclear_charge_mode == "per_species":
                raise ValueError(
                    "Specify effective_nuclear_charges for per_species mode"
                )
            effective_nuclear_charges = [0.0] * len(atomic_numbers)
        if len(effective_nuclear_charges) != len(atomic_numbers):
            raise ValueError("Supply one effective nuclear charge per species")
        hamiltonian_cutoff = (
            r_max if hamiltonian_cutoff is None else float(hamiltonian_cutoff)
        )
        if not np.isfinite(hamiltonian_cutoff) or not 0 < hamiltonian_cutoff <= r_max:
            raise ValueError(
                "hamiltonian_cutoff must be positive and no larger than r_max"
            )
        heads = heads or ["Default"]
        if atomic_energies is None:
            atomic_energies = np.zeros((len(heads), len(atomic_numbers)))
        self._init_local_model(
            r_max,
            num_bessel,
            num_polynomial_cutoff,
            max_ell,
            interaction_cls,
            interaction_cls_first,
            num_interactions,
            len(atomic_numbers),
            o3.Irreps(hidden_irreps),
            o3.Irreps(MLP_irreps),
            np.asarray(atomic_energies),
            avg_num_neighbors,
            atomic_numbers,
            correlation,
            gate,
            radial_MLP,
            radial_type,
            heads,
        )
        self.hamiltonian = HamiltonianBuilder(
            hidden_irreps,
            n_s,
            n_p,
            edge_mode,
            hamiltonian_cutoff,
            num_bessel,
            matrix_feature_multiplicity,
            matrix_radial_hidden,
            matrix_sh_max_l,
        )
        if self.spin_mode == "UKS":
            self.exchange_head = OnsiteHamiltonianHead(
                hidden_irreps, n_s, n_p, matrix_feature_multiplicity
            )
        if not math.isfinite(default_elec_temp) or default_elec_temp < 0:
            raise ValueError("default_elec_temp must be finite and nonnegative (eV)")
        self.register_buffer(
            "default_elec_temp", torch.tensor(float(default_elec_temp))
        )
        self.elec_temp_units = elec_temp_units
        self.electronic_state = ElectronicState(
            n_s + 3 * n_p,
            self.spin_mode,
            elec_temp_units,
            dos_grid=dos_grid,
            dos_width=dos_width,
            higher_order_backend=higher_order_backend,
            spectral_chunk_size=spectral_chunk_size,
        )
        if not math.isfinite(kspace_cutoff_factor) or kspace_cutoff_factor <= 0:
            raise ValueError("kspace_cutoff_factor must be positive and finite")
        cutoff = kspace_cutoff_factor * gto_basis_kspace_cutoff([sigma], 1)
        self.register_buffer("kspace_cutoff", torch.tensor(cutoff))
        self.pbc_handling = pbc_handling
        self.density_electrostatics = DensityElectrostatics(
            n_s,
            n_p,
            effective_nuclear_charges,
            sigma,
            pp_scalar,
            keep_quadrupoles,
            coupling_mode,
            C,
            nuclear_profile,
            nuclear_charge_mode,
            self_policy,
            pbc_handling,
            kspace_cutoff=cutoff,
        )
        self.energy_kind = energy_kind
        self.xc_type = xc_type
        xc_options = dict(
            readout=xc_readout,
            invariant_channels=xc_channels,
            hidden_features=xc_hidden,
        )
        if xc_type == "onsite":
            self.xc = OnsiteExchangeCorrelation(n_s, n_p, hidden_irreps, **xc_options)
        elif xc_type == "multipole":
            self.xc = MultipoleExchangeCorrelation(
                hidden_irreps, norm_epsilon=xc_norm_epsilon, **xc_options
            )
        elif xc_type == "none":
            self.xc = None
        else:
            raise ValueError("xc_type must be onsite, multipole or none")

    @torch.enable_grad()
    def forward(
        self,
        data,
        training=False,
        compute_force=True,
        compute_virials=False,
        compute_stress=False,
        return_density=False,
        return_electronic_state=False,
    ):
        if not isinstance(data, dict):
            data = data.to_dict()
        positions = data["positions"]
        if compute_force and not positions.requires_grad:
            positions = positions.clone().requires_grad_(True)
        batch = data.get(
            "batch",
            torch.zeros(len(positions), dtype=torch.long, device=positions.device),
        )
        ptr = data.get(
            "ptr", torch.tensor([0, len(positions)], device=positions.device)
        )
        ng = ptr.numel() - 1
        if bool((ptr[1:] <= ptr[:-1]).any()):
            raise ValueError("Empty graphs are not supported")
        expected_batch = torch.repeat_interleave(
            torch.arange(ng, device=batch.device), ptr[1:] - ptr[:-1]
        )
        if (
            int(ptr[0]) != 0
            or int(ptr[-1]) != len(positions)
            or not torch.equal(batch, expected_batch)
        ):
            raise ValueError("batch and ptr must describe contiguous, nonempty graphs")
        cell = data.get("cell", positions.new_zeros(ng, 3, 3)).reshape(ng, 3, 3)
        pbc = data.get(
            "pbc", torch.zeros(ng, 3, dtype=torch.bool, device=positions.device)
        ).reshape(ng, 3)
        displacement = positions.new_zeros(
            ng, 3, 3, requires_grad=compute_virials or compute_stress
        )
        strain = (displacement + displacement.mT) / 2
        x = positions + torch.einsum("ni,nij->nj", positions, strain[batch])
        cells = cell + cell @ strain
        edges = data["edge_index"]
        if "unit_shifts" in data:
            shifts = torch.einsum(
                "ni,nij->nj", data["unit_shifts"].to(x), cells[batch[edges[0]]]
            )
        else:
            raw = data.get("shifts", x.new_zeros(edges.shape[1], 3))
            shifts = raw + torch.einsum("ni,nij->nj", raw, strain[batch[edges[0]]])
        volume = torch.linalg.det(cells).abs()
        realspace = self.pbc_handling == "realspace" or (
            self.pbc_handling == "auto" and not bool(pbc.any())
        )
        if realspace:
            kv, kn, k0 = x.new_empty((0, 3)), x.new_empty(0), x.new_empty(0)
            kb = torch.empty(0, dtype=torch.long, device=x.device)
        else:
            if not bool((volume > 1e-10).all()):
                raise ValueError("Reciprocal electrostatics requires nonsingular cells")
            reciprocal = 2 * math.pi * torch.linalg.inv(cells.mT)
            kv, kn, kb, k0 = compute_k_vectors_flat(
                self.kspace_cutoff, cells, reciprocal
            )
        geometry = dict(
            positions=x,
            cell=cells,
            pbc=pbc,
            ptr=ptr,
            batch=batch,
            volume=volume,
            k_vectors=kv,
            k_norm2=kn,
            k_vector_batch=kb,
            k0_mask=k0,
        )
        attrs = data["node_attrs"]
        species = attrs.argmax(-1)

        def graph_field(name, default=None):
            value = data.get(name, default)
            if value is None:
                raise ValueError(f"Missing required graph field {name}")
            value = torch.as_tensor(value, device=x.device, dtype=x.dtype).reshape(-1)
            if value.numel() != ng:
                raise ValueError(f"{name} must contain one value per graph")
            return value

        alpha, beta = graph_field("N_alpha"), graph_field("N_beta")
        charge = graph_field("total_charge")
        if (
            not torch.isfinite(charge).all()
            or not data.get(
                "total_charge_present",
                torch.ones(ng, dtype=torch.bool, device=x.device),
            ).all()
        ):
            raise ValueError("total_charge must be explicitly supplied and finite")
        capacity = (ptr[1:] - ptr[:-1]) * self.electronic_state.block_size
        if (
            not (torch.isfinite(alpha).all() and torch.isfinite(beta).all())
            or (
                (alpha < 0)
                | (beta < 0)
                | (alpha > capacity)
                | (beta > capacity)
                | (alpha + beta > capacity)
            ).any()
        ):
            raise ValueError(
                "Electron counts must be nonnegative and N_alpha + N_beta <= basis size"
            )
        default_temperature = self.default_elec_temp.to(x).expand(ng)
        if self.elec_temp_units == "kelvin":
            default_temperature = default_temperature / 8.617333262145e-5
        temperature = graph_field("elec_temp", default_temperature)
        if "elec_temp_present" in data:
            temperature = torch.where(
                data["elec_temp_present"].reshape(-1), temperature, default_temperature
            )
        node_heads = data.get(
            "head", torch.zeros(ng, dtype=torch.long, device=x.device)
        ).reshape(-1)[batch]
        node_energy = (
            self.atomic_energies_fn(attrs).gather(1, node_heads[:, None]).squeeze(-1)
        )
        features = self.node_embedding(attrs)
        vectors = x[edges[1]] - x[edges[0]] + shifts
        lengths = vectors.norm(dim=-1, keepdim=True)
        edge_attrs = self.spherical_harmonics(vectors[:, [1, 2, 0]])
        edge_features, _ = self.radial_embedding(
            lengths, attrs, edges, self.atomic_numbers
        )
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            features, sc = interaction(
                node_attrs=attrs,
                node_feats=features,
                edge_attrs=edge_attrs,
                edge_feats=edge_features,
                edge_index=edges,
            )
            features = product(node_feats=features, sc=sc, node_attrs=attrs)
            node_energy = node_energy + readout(features, node_heads).gather(
                1, node_heads[:, None]
            ).squeeze(-1)
        local = x.new_zeros(ng).index_add(0, batch, node_energy)
        H = self.hamiltonian(features, x, edges, shifts, ptr)
        if self.spin_mode == "UKS":
            blocks = self.exchange_head(features)
            exchange = [
                torch.block_diag(*blocks[first:last].unbind())
                for first, last in zip(ptr[:-1].tolist(), ptr[1:].tolist())
            ]
            H = [(h + delta, h - delta) for h, delta in zip(H, exchange)]
        needs_derivatives = compute_force or compute_virials or compute_stress
        mode = (
            "second_order"
            if training and needs_derivatives
            else "first_order" if training or needs_derivatives else "none"
        )
        state = self.electronic_state(
            H,
            alpha,
            beta,
            temperature,
            derivative_mode=mode,
            return_full_density=return_electronic_state,
            return_eigenvectors=return_electronic_state,
        )
        result = self.density_electrostatics(
            state["gamma"],
            state["gamma_spin"],
            geometry,
            species,
            data.get("external_field"),
            data.get("effective_nuclear_charges"),
            return_density=return_density,
        )
        expected_charge = (
            x.new_zeros(ng).index_add(0, batch, result["effective_nuclear_charges"])
            - alpha
            - beta
        )
        if not torch.allclose(
            expected_charge, graph_field("total_charge"), atol=1e-5, rtol=0
        ):
            val = graph_field("total_charge")
            raise ValueError(
                f"N_alpha + N_beta is inconsistent with nuclear and total charge, {expected_charge}, {val}"
            )
        if self.xc_type == "onsite":
            atomic_xc = self.xc(state["gamma"], state["gamma_spin"], features)
        elif self.xc_type == "multipole":
            multipoles = result["multipoles"]
            atomic_xc = self.xc(
                multipoles["electron_alpha"] + multipoles["electron_beta"],
                multipoles["electron_alpha"] - multipoles["electron_beta"],
                features,
            )
        else:
            atomic_xc = features.new_zeros(features.shape[0])
        xc_energy = atomic_xc.new_zeros(ng).index_add(0, batch, atomic_xc)
        energy = (
            local + state["band_energy"] + result["electrostatic_energy"] + xc_energy
        )
        if self.energy_kind == "free":
            energy = energy + state["entropy_energy"]
        energy = energy + x.sum() * 0
        result.update(
            energy=energy,
            local_energy=local,
            xc_energy=xc_energy,
            atomic_xc_energy=atomic_xc,
            band_energy=state["band_energy"],
            entropy_energy=state["entropy_energy"],
            chemical_potentials=state["chemical_potentials"],
            count_residuals=state["count_residuals"],
            dos=state["dos"],
            cumulative_dos=state["cumulative_dos"],
            spectral_states=state["spectral_states"],
            dos_grid=self.electronic_state.dos_grid,
            forces=None,
            virials=None,
            stress=None,
        )
        targets = ([positions] if compute_force else []) + (
            [displacement] if compute_virials or compute_stress else []
        )
        if targets:
            grads = torch.autograd.grad(
                energy.sum(), targets, create_graph=training, retain_graph=True
            )
            if compute_force:
                result["forces"] = -grads[0]
            if compute_virials or compute_stress:
                result["virials"] = -grads[-1]
            if compute_stress:
                volume = torch.linalg.det(cell).abs()
                result["stress"] = torch.where(
                    (volume > 1e-10)[:, None, None],
                    grads[-1] / volume.clamp_min(1e-10)[:, None, None],
                    torch.zeros_like(grads[-1]),
                )
        if return_electronic_state:
            result.update(hamiltonians=H, **state)
        return result


class OneShotMLRKS(_OneShotMLBase):
    """One-shot restricted model: one Hamiltonian and equal spin populations."""

    spin_mode = "RKS"


class OneShotMLUKS(_OneShotMLBase):
    """One-shot unrestricted model with independent local exchange-like blocks."""

    spin_mode = "UKS"
