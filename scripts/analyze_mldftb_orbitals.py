"""Inspect exported MLDFTB orbitals and optionally check model covariance.

Example:
  PYTHONPATH=. python scripts/analyze_mldftb_orbitals.py \
    --directory examples/mldftb/trained_model --output /tmp/orbital_analysis \
    --model examples/mldftb/trained_model/positive/mldftb_rewrite_initial.model

Writes JSON statistics and frontier coefficient/localisation figures. Squared
coefficients are auxiliary-basis weights, not the mapped physical charge density.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read

from export_mldftb_orbitals import compute_orbitals


def load_orbitals(path):
    frames = read(path, ":")
    ns, np_ = frames[0].info["n_s"], frames[0].info["n_p"]
    coefficients = np.stack(
        [
            np.column_stack(
                [f.arrays[f"orbital_s_{s}"] for s in range(ns)]
                + [f.arrays[f"orbital_p_{p}"] for p in range(np_)]
            )
            for f in frames
        ]
    )  # orbital, atom, local basis
    energies = np.array([f.info["eigenvalue"] for f in frames])
    fillings = np.array(
        [[f.info["filling_alpha"], f.info["filling_beta"]] for f in frames]
    )
    return frames[0], energies, fillings, coefficients, ns, np_


def inspect_case(directory, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    atoms, energies, fillings, c, ns, np_ = load_orbitals(directory / "orbitals.xyz")
    weights = (c**2).sum(axis=2)
    count = int(round(fillings[:, 0].sum()))
    frontier = np.arange(max(0, count - 3), min(len(energies), count + 3))
    labels = [f"{a.symbol}{i}" for i, a in enumerate(atoms)]
    oxygen = np.flatnonzero(atoms.numbers == 8)
    membership = np.argmin(atoms.get_all_distances()[:, oxygen], axis=1)
    rows = []
    for i in frontier:
        rows.append(
            dict(
                index=int(i),
                energy=float(energies[i]),
                fillings=fillings[i].tolist(),
                effective_atoms=float(1 / np.sum(weights[i] ** 2)),
                s_weight=float(np.sum(c[i, :, :ns] ** 2)),
                atom_weights=weights[i].tolist(),
                oxygen_group_weights=[
                    float(weights[i, membership == k].sum()) for k in range(len(oxygen))
                ],
            )
        )
    # A gap and occupations together determine the relevant thermal response.
    mask = (fillings[:, 0] > 1e-8) & (fillings[:, 0] < 1 - 1e-8)
    tau = float(atoms.info["elec_temp"])
    inferred_mu = energies[mask] + tau * np.log(
        fillings[mask, 0] / (1 - fillings[mask, 0])
    )
    stats = dict(
        formula=atoms.get_chemical_formula(),
        atoms=len(atoms),
        orbitals=len(energies),
        counts=fillings.sum(axis=0).tolist(),
        tau_eV=tau,
        mu_alpha_eV=float(np.median(inferred_mu)),
        gap_eV=float(energies[count] - energies[count - 1]),
        orthogonality_error=float(
            np.max(
                np.abs(c.reshape(len(c), -1) @ c.reshape(len(c), -1).T - np.eye(len(c)))
            )
        ),
        occupation_susceptibility_per_eV=(fillings * (1 - fillings))
        .sum(axis=0)
        .tolist(),
        oxygen_indices=oxygen.tolist(),
        frontier=rows,
    )
    stats["occupation_susceptibility_per_eV"] = [
        v / tau for v in stats["occupation_susceptibility_per_eV"]
    ]
    fig, axes = plt.subplots(
        2, 1, figsize=(max(9, len(atoms) * 0.65), 7), constrained_layout=True
    )
    image = axes[0].imshow(
        weights[frontier], aspect="auto", vmin=0, vmax=1, cmap="viridis"
    )
    axes[0].set(
        xticks=np.arange(len(atoms)),
        xticklabels=labels,
        yticks=np.arange(len(frontier)),
        yticklabels=[f"{i}: {energies[i]:.4f} eV" for i in frontier],
        title=f"{directory.name}: frontier orbital weights per atom",
    )
    fig.colorbar(image, ax=axes[0], label="Sum of squared coefficients")
    components = c[frontier].reshape(len(frontier), -1)
    limit = abs(components).max()
    image = axes[1].imshow(
        components, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit
    )
    axes[1].set(
        xticks=np.arange(len(atoms)) * (ns + 3 * np_) + (ns + 3 * np_ - 1) / 2,
        xticklabels=labels,
        yticks=np.arange(len(frontier)),
        yticklabels=frontier,
        title="Signed coefficients within each atom: "
        + ", ".join(
            [f"s{s}" for s in range(ns)]
            + [f"p{p}{axis}" for p in range(np_) for axis in "xyz"]
        ),
    )
    for boundary in np.arange(1, len(atoms)) * (ns + 3 * np_) - 0.5:
        axes[1].axvline(boundary, color="0.4", linewidth=0.5)
    fig.colorbar(
        image, ax=axes[1], label="Orbital coefficient (arbitrary overall sign)"
    )
    fig.savefig(output / f"{directory.name}_frontier.png", dpi=180)
    plt.close(fig)
    # Geometry coloured by frontier weight; arrows show the p-shell coefficients.
    fig = plt.figure(figsize=(12, 5))
    for panel, i in enumerate((count - 1, count), 1):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        xyz = atoms.positions
        ax.scatter(
            *xyz.T,
            s=35 + 500 * weights[i],
            c=["tab:red" if z == 8 else "0.65" for z in atoms.numbers],
        )
        for a in range(len(atoms)):
            ax.text(*xyz[a], labels[a], fontsize=8)
            for b in range(a):
                if (
                    atoms.numbers[a] != atoms.numbers[b]
                    and np.linalg.norm(xyz[a] - xyz[b]) < 1.25
                ):
                    ax.plot(*xyz[[a, b]].T, color="0.6", linewidth=1)
        if np_:
            ax.quiver(*xyz.T, *c[i, :, ns : ns + 3].T, length=2, color="tab:blue")
        ax.set(
            title=f"Orbital {i}; alpha filling {fillings[i,0]:.4f}",
            xlabel="x (A)",
            ylabel="y (A)",
            zlabel="z (A)",
        )
        ax.set_box_aspect(np.maximum(np.ptp(xyz, axis=0), 0.5))
    fig.suptitle(
        f"{directory.name}: atom sizes show weights; arrows show first p-shell coefficients"
    )
    fig.tight_layout()
    fig.savefig(output / f"{directory.name}_geometry.png", dpi=180)
    plt.close(fig)
    return stats, atoms, energies, c, ns, np_


def covariance_checks(calculator, atoms, stored_values, stored_c, ns, np_):
    values, U, occ, mu, charge, _ = compute_orbitals(calculator, atoms)
    rng = np.random.default_rng(23)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rotation.T
    rotated.set_cell(atoms.cell.array @ rotation.T)
    if "external_field" in rotated.info:
        rotated.info["external_field"] = (
            np.asarray(rotated.info["external_field"]) @ rotation.T
        )
    re, ru, *_ = compute_orbitals(calculator, rotated)
    block = ns + 3 * np_
    expected = U.reshape(len(atoms), block, -1).copy()
    for p in range(np_):
        expected[:, ns + 3 * p : ns + 3 * p + 3] = np.einsum(
            "ab,nbk->nak", rotation, expected[:, ns + 3 * p : ns + 3 * p + 3]
        )
    expected = expected.reshape(U.shape)
    Hrot = (ru * re) @ ru.T
    Hexpected = (expected * values) @ expected.T
    reverse = np.arange(len(atoms))[::-1]
    pe, pu, *_ = compute_orbitals(calculator, atoms[reverse])
    pu = pu.reshape(len(atoms), block, -1)[np.argsort(reverse)].reshape(U.shape)
    H = (U * values) @ U.T
    # Same geometry at a different electron count: no electronic feedback in H.
    changed = atoms.copy()
    changed.info["N_alpha"] += 1
    changed.info["total_charge"] -= 1
    ce, *_ = compute_orbitals(calculator, changed)
    warm = atoms.copy()
    warm.info["elec_temp"] = 0.025
    we, _, wo, wm, _, _ = compute_orbitals(calculator, warm)
    return dict(
        total_charge=float(charge),
        reload_eigenvalue_error=float(abs(values - stored_values).max()),
        reload_hamiltonian_error=float(
            abs(
                (U * values) @ U.T
                - (stored_c.reshape(len(U), -1).T * stored_values)
                @ stored_c.reshape(len(U), -1)
            ).max()
        ),
        rotation_eigenvalue_error=float(abs(values - re).max()),
        rotation_hamiltonian_error=float(abs(Hrot - Hexpected).max()),
        rotation_min_matched_orbital_overlap=float(abs(np.diag(expected.T @ ru)).min()),
        permutation_hamiltonian_error=float(abs((pu * pe) @ pu.T - H).max()),
        changed_count_eigenvalue_error=float(abs(ce - values).max()),
        warmed_eigenvalue_error=float(abs(we - values).max()),
        warmed_occupations=wo.tolist(),
        warmed_mu=wm.tolist(),
    )


def ideal_water_probes(calculator):
    """Specific H/O probes using four auxiliary electrons per molecule."""
    from ase import Atoms

    results = {}
    for name in ("water", "hydronium"):
        if name == "water":
            angle = np.deg2rad(104.5 / 2)
            xyz = [
                [0, 0, 0],
                [0.97 * np.sin(angle), 0, 0.97 * np.cos(angle)],
                [-0.97 * np.sin(angle), 0, 0.97 * np.cos(angle)],
            ]
        else:
            radius = np.sqrt(0.98**2 - 0.35**2)
            xyz = [[0, 0, 0]] + [
                [radius * np.cos(t), radius * np.sin(t), 0.35]
                for t in [0, 2 * np.pi / 3, 4 * np.pi / 3]
            ]
        atoms = Atoms("OH2" if name == "water" else "OH3", positions=xyz, cell=[40] * 3)
        atoms.info.update(
            N_alpha=2,
            N_beta=2,
            elec_temp=0.0025,
            total_charge=0 if name == "water" else 1,
        )
        e, U, f, mu, _, _ = compute_orbitals(calculator, atoms)
        results[name] = dict(
            eigenvalues=e.tolist(), fillings=f.tolist(), mu=mu.tolist()
        )
        if name == "hydronium":
            angle = 2 * np.pi / 3
            R = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
            readout = calculator.model.hamiltonian.matrix.edge_readout
            ns, np_ = readout.n_s, readout.n_p
            c = U.reshape(4, ns + 3 * np_, -1).copy()
            for shell in range(np_):
                c[:, ns + 3 * shell : ns + 3 * shell + 3] = np.einsum(
                    "ab,nbk->nak", R, c[:, ns + 3 * shell : ns + 3 * shell + 3]
                )
            transformed = np.empty_like(c)
            transformed[[0, 2, 3, 1]] = c
            transformed = transformed.reshape(U.shape)
            pair_representation = U[:, 1:3].T @ transformed[:, 1:3]
            P = (U * f[:, 0]) @ U.T
            rotated_P = (transformed * f[:, 0]) @ transformed.T
            results[name]["C3_pair_trace"] = float(np.trace(pair_representation))
            results[name]["C3_pair_determinant"] = float(
                np.linalg.det(pair_representation)
            )
            results[name]["density_symmetry_error"] = float(abs(rotated_P - P).max())
            distorted = atoms.copy()
            distorted.positions[1] *= 1.01
            de, _, df, _, _, _ = compute_orbitals(calculator, distorted)
            results[name]["one_percent_bond_stretch_gap_eV"] = float(de[2] - de[1])
            results[name]["one_percent_bond_stretch_pair_fillings"] = df[1:3].tolist()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--ideal-water-probes", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    calculator = None
    if args.model:
        from mace_scf.calculators.mldftb import MLDFTBCalculator

        calculator = MLDFTBCalculator(model_path=args.model)
    results = {}
    for case in sorted(args.directory.iterdir()):
        if not (case / "orbitals.xyz").exists():
            continue
        stats, atoms, values, c, ns, np_ = inspect_case(case, args.output)
        if calculator:
            stats["checks"] = covariance_checks(calculator, atoms, values, c, ns, np_)
        results[case.name] = stats
    if args.ideal_water_probes:
        if calculator is None:
            parser.error("--ideal-water-probes requires --model")
        results["ideal_probes"] = ideal_water_probes(calculator)
    (args.output / "analysis.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
