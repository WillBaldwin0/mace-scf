"""ASE calculator for non-self-consistent MLDFTB models (eV and Angstrom).

ASE energy is the internal energy. free_energy is the selected model energy,
whose derivative gives forces/stress; use force_consistent=True when the saved
model uses energy_kind='free'. Electronic inputs are read from atoms.info and,
for per-atom nuclear charges, atoms.arrays. Changes to them invalidate caching.
"""

from copy import deepcopy

import numpy as np
from mace_scf.data.electronic_inputs import validate_configuration
import torch
from ase.calculators.calculator import (
    Calculator,
    all_changes,
    PropertyNotImplementedError,
)
from ase.stress import full_3x3_to_voigt_6_stress
from mace.data import KeySpecification, config_from_atoms
from mace.tools import AtomicNumberTable, torch_geometric

from mace_scf.data import ExtAtomicData
from mace_scf.electrostatics.mldftb import OneShotMLRKS, OneShotMLUKS


class MLDFTBCalculator(Calculator):
    implemented_properties = [
        "internal_energy",
        "model_energy",
        "multipoles",
        "spectrum",
        "electronic_state",
        "dos",
        "cumulative_dos",
        "dos_grid",
        "gap",
        "energy",
        "free_energy",
        "forces",
        "stress",
        "charges",
        "dipole",
        "density_coefficients",
        "atomic_dipoles",
        "atomic_spin",
        "local_energy",
        "band_energy",
        "xc_energy",
        "electrostatic_energy",
        "entropy_energy",
        "electronic_free_energy",
        "chemical_potentials",
    ]

    def __init__(
        self,
        model_path=None,
        *,
        model=None,
        device="cpu",
        default_dtype="float64",
        skip_projected_spin_check=False,
        atomic_spin_key="atomic_spin",
        atomic_multipoles_key="atomic_multipoles",
        N_alpha_key="N_alpha",
        N_beta_key="N_beta",
        elec_temp_key="elec_temp",
        total_charge_key="total_charge",
        external_field_key="external_field",
        effective_nuclear_charges_key="effective_nuclear_charges",
        head=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if (model_path is None) == (model is None):
            raise ValueError("Supply exactly one of model_path or model")
        if default_dtype not in {"float32", "float64"}:
            raise ValueError("default_dtype must be float32 or float64")
        self.model_path = str(model_path) if model_path is not None else None
        self.device = torch.device(device)
        self.dtype = getattr(torch, default_dtype)
        self.model = (
            torch.load(model_path, map_location=self.device, weights_only=False)
            if model is None
            else deepcopy(model)
        )
        if not isinstance(self.model, (OneShotMLRKS, OneShotMLUKS)):
            raise TypeError(
                "MLDFTBCalculator requires a OneShotMLRKS or OneShotMLUKS model"
            )
        self.model.to(device=self.device, dtype=self.dtype).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.z_table = AtomicNumberTable(self.model.atomic_numbers.cpu().tolist())
        self.head = self.model.heads[0] if head is None else head
        if self.head not in self.model.heads:
            raise ValueError(f"Unknown model head {self.head!r}")
        self.info_keys = dict(
            N_alpha=N_alpha_key,
            N_beta=N_beta_key,
            elec_temp=elec_temp_key,
            total_charge=total_charge_key,
            external_field=external_field_key,
        )
        self.skip_projected_spin_check = skip_projected_spin_check
        self.nuclear_key = effective_nuclear_charges_key
        self.keyspec = KeySpecification(
            info_keys=self.info_keys,
            arrays_keys={
                "effective_nuclear_charges": self.nuclear_key,
                "atomic_spin": atomic_spin_key,
                "atomic_multipoles": atomic_multipoles_key,
            },
        )

    @property
    def basis_metadata(self):
        density = self.model.density_electrostatics.density_map.shell_coupling
        # The Hamiltonian layout is s shells followed by Cartesian p shells.
        return dict(
            n_s=density.n_s,
            n_p=density.n_p,
            block_size=self.model.electronic_state.block_size,
            orbital_order="atom, s shells, p shells (x,y,z)",
            eigenvectors="columns",
            multipole_order=["q", "dy", "dz", "dx"],
            phase="Condon-Shortley",
        )

    def _ensure(self, atoms, properties):
        properties = list(properties)
        unknown = set(properties) - set(self.implemented_properties)
        if unknown:
            raise PropertyNotImplementedError(str(sorted(unknown)))
        changes = self.check_state(atoms)
        if changes or any(key not in self.results for key in properties):
            self.calculate(atoms, properties, changes)

    def evaluate(self, atoms, properties=("energy",)):
        self._ensure(atoms, properties)
        return deepcopy({key: self.results[key] for key in properties})

    def get_electronic_state(self, atoms):
        return self.evaluate(atoms, ["electronic_state"])["electronic_state"]

    def get_state_multipoles(self, atoms, channel=None, indices=None):
        self._ensure(atoms, ["electronic_state"])
        state = self.results["electronic_state"]
        if channel is None:
            channel = "shared" if self.model.spin_mode == "RKS" else "alpha"
        coefficients = state["channels"][channel]["eigenvectors"]
        selected = (
            np.arange(coefficients.shape[1])
            if indices is None
            else np.asarray(indices, dtype=int)
        )
        if (
            selected.ndim != 1
            or ((selected < 0) | (selected >= coefficients.shape[1])).any()
        ):
            raise ValueError("State indices must be a one-dimensional array in range")
        result = []
        with torch.no_grad():
            for index in selected:
                local = torch.as_tensor(
                    coefficients[:, index].copy(), device=self.device, dtype=self.dtype
                ).reshape(len(atoms), -1)
                blocks = local[:, :, None] * local[:, None, :]
                result.append(
                    self.model.density_electrostatics.density_map(blocks).cpu().numpy()
                )
        return np.stack(result) if result else np.empty((0, len(atoms), 4))

    def check_state(self, atoms, tol=1e-15):
        changes = super().check_state(atoms, tol=tol)
        if getattr(self, "_cached_head", self.head) != self.head:
            changes.append("head")
        if self.atoms is not None:
            for key in self.info_keys.values():
                old, new = self.atoms.info.get(key), atoms.info.get(key)
                if not np.array_equal(old, new):
                    changes.append(key)
            for key in self.keyspec.arrays_keys.values():
                if not np.array_equal(
                    self.atoms.arrays.get(key), atoms.arrays.get(key)
                ):
                    changes.append(key)
        return changes

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        properties = set(properties or ["energy"])
        previous = {} if system_changes else self.results.copy()
        if "stress" in properties and abs(atoms.cell.volume) <= 1e-10:
            raise PropertyNotImplementedError("Stress requires a nonsingular cell")
        super().calculate(atoms, properties, system_changes)
        self._cached_head = self.head
        # ASE copies info shallowly; retain a snapshot of mutable field arrays.
        self.atoms.info = deepcopy(self.atoms.info)
        for name in ("N_alpha", "N_beta", "total_charge"):
            if self.info_keys[name] not in self.atoms.info:
                raise ValueError(
                    f"Missing atoms.info[{self.info_keys[name]!r}] ({name})"
                )
        config = config_from_atoms(
            self.atoms, key_specification=self.keyspec, head_name=self.head
        )
        nuclei = self.model.density_electrostatics.nuclei
        validate_configuration(
            config,
            {
                "n_s": self.model.electronic_state.block_size,
                "n_p": 0,
                "nuclear_charge_mode": nuclei.charge_mode,
                "effective_nuclear_charges": dict(
                    zip(
                        self.z_table.zs,
                        nuclei.effective_charges.detach().cpu().tolist(),
                    )
                ),
            },
            self.model.spin_mode,
            self.skip_projected_spin_check,
        )
        # ExtAtomicData uses the global default dtype when building the graph.
        # Restore it afterwards; all graph floats now match the model dtype.
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(self.dtype)
            data = ExtAtomicData.from_config(
                config,
                self.z_table,
                float(self.model.r_max),
                heads=self.model.heads,
                atomic_multipoles_max_l=1,
                preserve_cell=True,
            )
        finally:
            torch.set_default_dtype(previous_dtype)
        batch = next(
            iter(torch_geometric.dataloader.DataLoader([data], batch_size=1))
        ).to(self.device)
        output = self.model(
            batch,
            compute_force="forces" in properties,
            compute_stress="stress" in properties,
            return_electronic_state="electronic_state" in properties,
        )

        def array(key):
            return output[key].detach().cpu().numpy()

        def scalar(key):
            return float(array(key).reshape(()))

        internal = (
            scalar("local_energy")
            + scalar("band_energy")
            + scalar("electrostatic_energy")
            + scalar("xc_energy")
        )
        self.results = dict(
            energy=internal,
            free_energy=scalar("energy"),
            electronic_free_energy=internal + scalar("entropy_energy"),
            charges=array("atomic_charges"),
            dipole=array("dipole")[0],
            density_coefficients=array("density_coefficients"),
            atomic_dipoles=array("atomic_dipoles"),
            atomic_spin=array("atomic_spin"),
            chemical_potentials=array("chemical_potentials")[0],
            **{
                key: scalar(key)
                for key in (
                    "local_energy",
                    "band_energy",
                    "xc_energy",
                    "electrostatic_energy",
                    "entropy_energy",
                )
            },
            dos=array("dos")[0],
            cumulative_dos=array("cumulative_dos")[0],
            dos_grid=array("dos_grid"),
        )

        self.results.update(
            internal_energy=internal,
            model_energy=scalar("energy"),
            multipoles={
                k: v.detach().cpu().numpy() for k, v in output["multipoles"].items()
            },
        )
        channels = {}
        labels = ["shared"] if self.model.spin_mode == "RKS" else ["alpha", "beta"]
        for label, raw in zip(labels, output["spectral_states"][0]):
            channels[label] = {
                k: v.detach().cpu().numpy().copy() for k, v in raw.items()
            }
        spectrum = dict(
            spin_mode=self.model.spin_mode,
            channels={
                label: {
                    k: v
                    for k, v in values.items()
                    if k not in {"density", "eigenvectors"}
                }
                for label, values in channels.items()
            },
        )
        if self.model.spin_mode == "RKS":
            shared = spectrum["channels"]["shared"]
            shared["occupations_alpha"] = shared["occupations"].copy()
            shared["occupations_beta"] = shared["occupations"].copy()
        self.results["spectrum"] = spectrum
        from mace_scf.analysis.spectra import gap_summary

        self.results["gap"] = gap_summary(spectrum)
        if "electronic_state" in properties:
            matrices = output["hamiltonians"][0]
            if self.model.spin_mode == "RKS":
                matrices = [matrices]
            for label, matrix in zip(labels, matrices):
                channels[label]["hamiltonian"] = matrix.detach().cpu().numpy().copy()
            self.results["electronic_state"] = dict(
                spin_mode=self.model.spin_mode,
                channels=channels,
                basis=self.basis_metadata,
                onsite_density=array("gamma"),
                onsite_spin_density=array("gamma_spin"),
            )
        if output["forces"] is not None:
            self.results["forces"] = array("forces")
        if output["stress"] is not None:
            self.results["stress"] = full_3x3_to_voigt_6_stress(array("stress")[0])
        self.results = {**previous, **self.results}
