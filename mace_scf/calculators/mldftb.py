"""ASE calculator for non-self-consistent MLDFTB models (eV and Angstrom).

ASE energy is the internal energy. free_energy is the selected model energy,
whose derivative gives forces/stress; use force_consistent=True when the saved
model uses energy_kind='free'. Electronic inputs are read from atoms.info and,
for per-atom nuclear charges, atoms.arrays. Changes to them invalidate caching.
"""

from copy import deepcopy

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from mace.data import KeySpecification, config_from_atoms
from mace.tools import AtomicNumberTable, torch_geometric

from mace_scf.data import ExtAtomicData
from mace_scf.electrostatics.mldftb import MLDFTB


class MLDFTBCalculator(Calculator):
    implemented_properties = [
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
        self.device = torch.device(device)
        self.dtype = getattr(torch, default_dtype)
        self.model = (
            torch.load(model_path, map_location=self.device, weights_only=False)
            if model is None
            else deepcopy(model)
        )
        if not isinstance(self.model, MLDFTB):
            raise TypeError("MLDFTBCalculator requires an MLDFTB model")
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
        self.nuclear_key = effective_nuclear_charges_key
        self.keyspec = KeySpecification(
            info_keys=self.info_keys,
            arrays_keys={"effective_nuclear_charges": self.nuclear_key},
        )

    def check_state(self, atoms, tol=1e-15):
        changes = super().check_state(atoms, tol=tol)
        if self.atoms is not None:
            for key in self.info_keys.values():
                old, new = self.atoms.info.get(key), atoms.info.get(key)
                if not np.array_equal(old, new):
                    changes.append(key)
            if not np.array_equal(
                self.atoms.arrays.get(self.nuclear_key),
                atoms.arrays.get(self.nuclear_key),
            ):
                changes.append(self.nuclear_key)
        return changes

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        # ASE copies info shallowly; retain a snapshot of mutable field arrays.
        self.atoms.info = deepcopy(self.atoms.info)
        for name in ("N_alpha", "N_beta"):
            if self.info_keys[name] not in self.atoms.info:
                raise ValueError(
                    f"Missing atoms.info[{self.info_keys[name]!r}] ({name})"
                )
        config = config_from_atoms(
            self.atoms, key_specification=self.keyspec, head_name=self.head
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
        output = self.model(batch, compute_stress=bool(self.atoms.cell.volume > 1e-10))

        def array(key):
            return output[key].detach().cpu().numpy()

        def scalar(key):
            return float(array(key).reshape(()))

        internal = (
            scalar("local_energy")
            + scalar("band_energy")
            + scalar("electrostatic_energy")
        )
        self.results = dict(
            energy=internal,
            free_energy=scalar("energy"),
            electronic_free_energy=internal + scalar("entropy_energy"),
            forces=array("forces"),
            stress=(
                full_3x3_to_voigt_6_stress(array("stress")[0])
                if output["stress"] is not None
                else np.zeros(6)
            ),
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
                    "electrostatic_energy",
                    "entropy_energy",
                )
            },
        )
