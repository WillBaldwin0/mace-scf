"""Shared command-line input handling and explicitly named XYZ outputs."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from ase.io import iread
from mace_scf.calculators.mldftb import MLDFTBCalculator


def parser(description, overwrite=True):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", required=True)
    p.add_argument("--input", "--xyz", dest="input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--index",
        default=":",
        help="Zero-based index or nonnegative start:stop:step slice",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--head")
    if overwrite:
        p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-errors", action="store_true")
    p.add_argument("--skip-projected-spin-check", action="store_true")
    for key in (
        "N_alpha",
        "N_beta",
        "total_charge",
        "elec_temp",
        "external_field",
        "effective_nuclear_charges",
        "atomic_spin",
        "atomic_multipoles",
    ):
        p.add_argument("--" + key + "-key", default=key)
    for key in ("N_alpha", "N_beta", "total_charge", "elec_temp"):
        p.add_argument("--" + key, type=float, default=None)
    p.add_argument("--external-field", type=float, nargs=3)
    return p


def calculator(args):
    kwargs = {key: value for key, value in vars(args).items() if key.endswith("_key")}
    return MLDFTBCalculator(
        args.model,
        device=args.device,
        default_dtype=args.dtype,
        head=args.head,
        skip_projected_spin_check=args.skip_projected_spin_check,
        **kwargs,
    )


def frames(args):
    parts = args.index.split(":")
    if len(parts) == 1:
        start = int(parts[0])
        stop, step = start + 1, 1
    else:
        if len(parts) > 3:
            raise ValueError("Use an index or start:stop:step")
        parts += [""] * (3 - len(parts))
        start, stop, step = (
            int(parts[0] or 0),
            int(parts[1]) if parts[1] else None,
            int(parts[2] or 1),
        )
    if start < 0 or (stop is not None and stop < 0) or step <= 0:
        raise ValueError(
            "Streaming selection requires nonnegative indices and positive step"
        )
    for i, source in enumerate(iread(args.input, index=":")):
        if stop is not None and i >= stop:
            break
        if i < start or (i - start) % step:
            continue
        atoms = source.copy()
        atoms.info = deepcopy(source.info)
        # Preserve reference SinglePointCalculator properties explicitly.
        if source.calc is not None:
            for key, value in source.calc.results.items():
                target = (
                    atoms.arrays
                    if key in {"forces", "charges", "magmoms", "energies", "stresses"}
                    else atoms.info
                )
                if key not in target:
                    target[key] = deepcopy(value)
        for key in ("N_alpha", "N_beta", "total_charge", "elec_temp"):
            value = getattr(args, key)
            if value is not None:
                mapped = getattr(args, key + "_key")
                if mapped in atoms.info:
                    atoms.info["source_" + mapped] = deepcopy(atoms.info[mapped])
                atoms.info[mapped] = value
        if args.external_field is not None:
            atoms.info["source_" + args.external_field_key] = deepcopy(
                atoms.info.get(args.external_field_key, np.zeros(3))
            )
            atoms.info[args.external_field_key] = np.array(args.external_field)
        yield i, atoms


def json_write(path, values):
    def convert(value):
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [convert(v) for v in value]
        if isinstance(value, np.ndarray):
            return convert(value.tolist())
        if isinstance(value, np.generic):
            return convert(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    Path(path).write_text(json.dumps(convert(values), indent=2, allow_nan=False))


ATOMIC = {"forces", "charges", "atomic_spin", "density_coefficients", "atomic_dipoles"}


def add_predictions(atoms, result, prefix="pred_", overwrite=False):
    def put(name, value, atomic=False):
        key = prefix + name
        if not overwrite and (key in atoms.info or key in atoms.arrays):
            raise ValueError(f"Output key collision: {key}")
        if atomic:
            atoms.set_array(key, np.array(value))
        else:
            atoms.info[key] = value

    for name, value in result.items():
        if name == "spectrum":
            put("spin_mode", value["spin_mode"])
            for channel, c in value["channels"].items():
                for k in (
                    "eigenvalues",
                    "occupations",
                    "occupations_alpha",
                    "occupations_beta",
                ):
                    if k not in c:
                        continue
                    # Lists serialize as JSON metadata and retain singleton shape.
                    put(channel + "_" + k, c[k].tolist())
        elif name == "chemical_potentials":
            for spin, mu in zip(("alpha", "beta"), value):
                put("mu_" + spin, float(mu))
        elif name == "multipoles":
            for channel, coefficients in value.items():
                put("multipoles_" + channel, coefficients, True)
        elif name == "gap":
            for k, v in value.items():
                put("gap_" + k, v)
        else:
            put(name, value, name in ATOMIC)
