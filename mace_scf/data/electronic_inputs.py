"""Input policies for the one-shot electronic models (not eigensolver limits)."""

import numpy as np


def validate_configuration(config, options, spin_mode, skip_projected_spin_check=False):
    p = config.properties
    values = {}
    for key in ("N_alpha", "N_beta", "total_charge"):
        value = p.get(key)
        if value is None or np.asarray(value).size != 1 or not np.isfinite(value).all():
            raise ValueError(f"{key} must be supplied as a finite scalar")
        values[key] = float(value)
    a, b, q = (values[k] for k in ("N_alpha", "N_beta", "total_charge"))
    capacity = len(config.atomic_numbers) * (
        options.get("n_s", 1) + 3 * options.get("n_p", 1)
    )
    if not (0 <= a <= capacity and 0 <= b <= capacity and a + b <= capacity):
        raise ValueError(
            f"Electron counts must be nonnegative and N_alpha + N_beta <= basis size ({capacity})"
        )
    if spin_mode == "UKS" and b > a:
        raise ValueError("UKS requires N_beta <= N_alpha")
    if spin_mode == "RKS" and a != b:
        raise ValueError("RKS requires N_alpha == N_beta")
    if options.get("nuclear_charge_mode", "per_species") == "from_data":
        charges = p.get("effective_nuclear_charges")
        if charges is None or np.asarray(charges).shape != (
            len(config.atomic_numbers),
        ):
            raise ValueError(
                "effective_nuclear_charges must contain one value per atom"
            )
    else:
        mapping = options.get("effective_nuclear_charges")
        if mapping is None:
            raise ValueError("Specify effective_nuclear_charges for per_species mode")
        charges = [mapping[int(z)] for z in config.atomic_numbers]
    if not np.isfinite(charges).all() or abs(np.sum(charges) - a - b - q) > 1e-5:
        raise ValueError(
            "Electron counts, nuclear charges and total_charge are inconsistent"
        )
    temperature = p.get("elec_temp")
    if temperature is not None and (
        np.asarray(temperature).size != 1
        or not np.isfinite(temperature).all()
        or float(temperature) < 0
    ):
        raise ValueError("elec_temp must be a finite nonnegative scalar")
    multipoles = p.get("atomic_multipoles")
    if multipoles is not None:
        multipoles = np.asarray(multipoles)
        if (
            multipoles.ndim != 2
            or multipoles.shape[0] != len(config.atomic_numbers)
            or multipoles.shape[1] < 4
        ):
            raise ValueError(
                "atomic_multipoles requires at least four columns per atom"
            )
        monopoles = multipoles[:, 0]
        if not np.isfinite(monopoles).all() or abs(monopoles.sum() - q) > 1e-5:
            raise ValueError("Reference monopole sum disagrees with total_charge")
    spin = p.get("atomic_spin")
    if spin is not None and not skip_projected_spin_check:
        spin = np.asarray(spin)
        if (
            spin.shape != (len(config.atomic_numbers),)
            or not np.isfinite(spin).all()
            or abs(spin.sum() - (a - b)) > 1e-3
        ):
            raise ValueError(
                "Reference projected spin sum disagrees with N_alpha - N_beta"
            )
