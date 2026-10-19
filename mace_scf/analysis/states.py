"""State export using only public calculator methods."""

from copy import deepcopy
import numpy as np
from ase import Atoms
from ase.io import write
from .io import json_write
from .spectra import plot_spectrum


def export_configuration(
    calc,
    atoms,
    directory,
    *,
    indices=None,
    energy_window=None,
    plot=False,
    width=0.2,
    coefficients=False,
    matrices=False,
    multipoles_npz=False,
    coefficient_arrays=False,
    source_index=0,
):
    result = calc.evaluate(
        atoms, ["electronic_state", "spectrum", "dos", "cumulative_dos", "dos_grid"]
    )
    state, spectrum = result["electronic_state"], result["spectrum"]
    basis = state["basis"]
    ns, np_ = basis["n_s"], basis["n_p"]
    write(directory / "structure.xyz", atoms, format="extxyz", write_results=False)
    spectral_arrays = {k: result[k] for k in ("dos", "cumulative_dos", "dos_grid")}
    selections = {}
    for channel, c in state["channels"].items():
        order = np.argsort(c["eigenvalues"], kind="stable")
        chosen = order if indices is None else np.array(indices, dtype=int)
        if chosen.ndim != 1 or ((chosen < 0) | (chosen >= len(order))).any():
            raise ValueError("Orbital indices outside spectrum")
        if len(np.unique(chosen)) != len(chosen):
            raise ValueError("Repeated orbital indices")
        chosen = chosen[np.argsort(c["eigenvalues"][chosen], kind="stable")]
        if energy_window is not None:
            lo, hi = energy_window
            if lo > hi:
                raise ValueError("Energy window must be increasing")
            chosen = chosen[
                (c["eigenvalues"][chosen] >= lo) & (c["eigenvalues"][chosen] <= hi)
            ]
        selections[channel] = chosen.tolist()
        spectral_arrays.update(
            {
                channel + "_" + k: c[k]
                for k in ("eigenvalues", "occupations", "chemical_potential")
            }
        )
        filename = {
            "shared": "orbitals.xyz",
            "alpha": "orbitals_up.xyz",
            "beta": "orbitals_down.xyz",
        }[channel]
        # One state at a time: output size does not become peak working memory.
        collected = []
        with (directory / filename).open("w") as stream:
            for k in chosen:
                density = calc.get_state_multipoles(atoms, channel, [int(k)])[0]
                if multipoles_npz:
                    collected.append(density)
                frame = Atoms(
                    numbers=atoms.numbers,
                    positions=atoms.positions,
                    cell=atoms.cell,
                    pbc=atoms.pbc,
                )
                f = float(c["occupations"][k])
                fa = f if channel in ("shared", "alpha") else 0.0
                fb = f if channel in ("shared", "beta") else 0.0
                frame.info.update(
                    eigenvalue=float(c["eigenvalues"][k]),
                    orbital_index=int(k),
                    spin_channel=channel,
                    occupancy=fa + fb,
                    occupancy_alpha=fa,
                    occupancy_beta=fb,
                    density_normalization=1.0,
                    density_kind="electron_number",
                    multipoles_include_nuclei=False,
                    occupation_weighted=False,
                    parent_total_charge=float(
                        atoms.info[calc.info_keys["total_charge"]]
                    ),
                    parent_N_alpha=float(atoms.info[calc.info_keys["N_alpha"]]),
                    parent_N_beta=float(atoms.info[calc.info_keys["N_beta"]]),
                    source_frame=source_index,
                    n_s=ns,
                    n_p=np_,
                )
                frame.set_array("density_coefficients", density)
                if coefficient_arrays:
                    local = c["eigenvectors"][:, k].reshape(len(atoms), -1)
                    for j in range(ns):
                        frame.set_array(f"orbital_s_{j}", local[:, j].copy())
                    for j in range(np_):
                        frame.set_array(
                            f"orbital_p_{j}",
                            local[:, ns + 3 * j : ns + 3 * j + 3].copy(),
                        )
                write(stream, frame, format="extxyz", write_results=False)
        if multipoles_npz:
            np.savez(
                directory / (channel + "_state_multipoles.npz"),
                indices=chosen,
                density_coefficients=(
                    np.stack(collected) if collected else np.empty((0, len(atoms), 4))
                ),
            )
        if coefficients:
            np.savez(
                directory / (channel + "_states.npz"),
                indices=chosen,
                eigenvectors=c["eigenvectors"][:, chosen],
                n_s=ns,
                n_p=np_,
                atomic_numbers=atoms.numbers,
            )
        if matrices:
            np.savez(
                directory / (channel + "_matrices.npz"),
                hamiltonian=c["hamiltonian"],
                density=c["density"],
            )
    if plot:
        spectral_arrays.update(
            {
                "display_" + k: v
                for k, v in plot_spectrum(
                    directory / "spectrum.png",
                    spectrum,
                    float(atoms.info[calc.info_keys["total_charge"]]),
                    width,
                ).items()
            }
        )
    np.savez(directory / "spectrum.npz", **spectral_arrays)
    json_write(
        directory / "manifest.json",
        dict(
            source_frame=source_index,
            model=calc.model_path,
            spin_mode=state["spin_mode"],
            basis=basis,
            exported_indices=selections,
            density_kind="electron_number",
            density_normalization=1.0,
            energy_units="eV",
            multipole_units="electron * Angstrom^l",
            inputs={
                key: deepcopy(atoms.info.get(mapped))
                for key, mapped in calc.info_keys.items()
            },
            default_elec_temp_eV=float(calc.model.default_elec_temp),
            elec_temp_units=calc.model.elec_temp_units,
            display_width_eV=width if plot else None,
        ),
    )
