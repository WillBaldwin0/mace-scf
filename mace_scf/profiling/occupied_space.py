"""Benchmark sparse occupied-space solvers; see ml_dftb/occupied_space_benchmark.md.

Run with: python -m mace_scf.profiling.occupied_space --help
No force, electrostatic or SCF calculations are performed.
"""

import argparse
import csv
import json
import math
import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from e3nn import o3
from mace.data import config_from_atoms
from mace.tools import torch_geometric

from mace_scf.calculators.mldftb import MLDFTBCalculator
from mace_scf.data import ExtAtomicData
from mace_scf.electrostatics.matrix_ops import _fill
from mace_scf.profiling.polynomial import chebyshev, polynomial_density


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device, function):
    synchronize(device)
    start = time.perf_counter()
    result = function()
    synchronize(device)
    return result, 1000 * (time.perf_counter() - start)


class BlockHamiltonian:
    """Symmetric, coalesced atom-pair blocks, including periodic self images.

    Matvecs use either chunked edge gather/bmm/scatter or scalar CSR sparse.mm.
    Neither backend allocates a dense M x M array. Dense materialisation is an
    explicit, separately timed operation used only by the reference benchmark.
    """

    def __init__(self, edge_blocks, edge_index, onsite, edge_chunk=1024):
        self.n, self.b = onsite.shape[:2]
        self.size = self.n * self.b
        self.device, self.dtype = onsite.device, onsite.dtype
        self.edge_chunk = edge_chunk
        i, j = edge_index
        nodes = torch.arange(self.n, device=self.device)
        keys = torch.cat((i * self.n + j, j * self.n + i, nodes * (self.n + 1)))
        values = torch.cat(
            (edge_blocks / 2, edge_blocks.mT / 2, (onsite + onsite.mT) / 2)
        )
        unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        self.blocks = onsite.new_zeros(len(unique), self.b, self.b).index_add_(
            0, inverse, values
        )
        self.rows, self.cols = unique // self.n, unique % self.n
        self.diagonal_blocks = self.blocks[self.rows == self.cols]
        self.diagonal = self.diagonal_blocks.diagonal(dim1=-2, dim2=-1).reshape(-1)
        row_sum = onsite.new_zeros(self.n, self.b).index_add_(
            0, self.rows, self.blocks.abs().sum(-1)
        )
        self.lower_bound = (
            self.diagonal - (row_sum.reshape(-1) - self.diagonal.abs())
        ).min()
        self.csr = None
        self.backend = "edges"
        self.calls = self.vector_products = 0

    def prepare(self, backend):
        if backend == "edges":
            self.csr = None
        if backend == "csr" and self.csr is None:
            local = torch.arange(self.b, device=self.device)
            rows = (self.rows[:, None, None] * self.b + local[None, :, None]).expand(
                -1, self.b, self.b
            )
            cols = (self.cols[:, None, None] * self.b + local[None, None, :]).expand(
                -1, self.b, self.b
            )
            coo = torch.sparse_coo_tensor(
                torch.stack((rows.flatten(), cols.flatten())),
                self.blocks.flatten(),
                (self.size, self.size),
            ).coalesce()
            self.csr = coo.to_sparse_csr()
        self.backend = backend

    def __call__(self, x):
        if x.shape[1] == 0:
            return torch.zeros_like(x)
        self.calls += 1
        self.vector_products += x.shape[1]
        if self.backend == "csr":
            return torch.sparse.mm(self.csr, x)
        shaped = x.reshape(self.n, self.b, -1)
        result = torch.zeros_like(shaped)
        for first in range(0, len(self.blocks), self.edge_chunk):
            last = first + self.edge_chunk
            product = torch.bmm(self.blocks[first:last], shaped[self.cols[first:last]])
            result.index_add_(0, self.rows[first:last], product)
        return result.reshape(self.size, -1)

    def dense(self):
        result = self.blocks.new_zeros(self.n, self.n, self.b, self.b)
        result[self.rows, self.cols] = self.blocks
        return result.permute(0, 2, 1, 3).reshape(self.size, self.size)


def node_features(model, data):
    """The production MACE feature path, without energy readouts/electrostatics."""
    attrs, positions, edges = data["node_attrs"], data["positions"], data["edge_index"]
    shifts = data["shifts"]
    vectors = positions[edges[1]] - positions[edges[0]] + shifts
    lengths = vectors.norm(dim=-1, keepdim=True)
    edge_attrs = model.spherical_harmonics(vectors[:, [1, 2, 0]])
    radial, _ = model.radial_embedding(lengths, attrs, edges, model.atomic_numbers)
    features = model.node_embedding(attrs)
    for interaction, product in zip(model.interactions, model.products):
        features, sc = interaction(
            node_attrs=attrs,
            node_feats=features,
            edge_attrs=edge_attrs,
            edge_feats=radial,
            edge_index=edges,
        )
        features = product(node_feats=features, sc=sc, node_attrs=attrs)
    return features


def hamiltonian_blocks(builder, features, positions, edge_index, shifts):
    """Mirror HamiltonianBuilder up to assembly, avoiding its dense allocation.

    Tests compare these blocks with the production builder for both heads,
    on-site modes, shortened cutoffs, duplicate image pairs and empty edge sets.
    """
    sender, receiver = edge_index
    vectors = positions[receiver] - positions[sender] + shifts
    distances = vectors.norm(dim=-1)
    if bool((distances == 0).any()):
        raise ValueError("Zero-length Hamiltonian edge")
    mask = distances < builder.r_max
    edges = edge_index[:, mask]
    sender, receiver = edges
    vectors, distances = vectors[mask], distances[mask]
    sh = o3.spherical_harmonics(
        builder.sh_irreps,
        vectors[:, [1, 2, 0]],
        normalize=True,
        normalization="component",
    )
    radial = torch.exp(
        -(((distances[:, None] - builder.radial_centres) / builder.r_max) ** 2)
    )
    x = (distances / builder.r_max).clamp(0, 1)
    envelope = 1 - 10 * x**3 + 15 * x**4 - 6 * x**5
    if builder.edge_mode == "bilinear":
        pair = builder.pair_linear(
            builder.pair_tp(features[sender], features[receiver])
        )
    else:
        pair = (
            builder.sender_linear(features)[sender]
            + builder.receiver_linear(features)[receiver]
        )
    edge_features = (
        builder.angular_tp(pair, sh, builder.radial_mlp(radial)) * envelope[:, None]
    )
    onsite_features = (
        builder.onsite_tp(builder.onsite_left(features), builder.onsite_right(features))
        if builder.onsite_mode == "quadratic"
        else features
    )
    blocks = builder.matrix.edge_readout(edge_features)
    onsite = builder.matrix.onsite_readout(onsite_features)
    order = builder.orbital_order
    return blocks[:, order][:, :, order], edges, onsite[:, order][:, :, order]


class Preconditioner:
    """Diagonal Davidson correction or fixed SPD shifted Jacobi/block Jacobi."""

    def __init__(self, operator, kind):
        self.op, self.kind = operator, kind
        guard = max(1e-3, float(operator.diagonal.abs().max()) * 1e-3)
        self.shift = operator.lower_bound - guard
        self.cholesky = None
        if kind == "block":
            eye = torch.eye(operator.b, device=operator.device, dtype=operator.dtype)
            self.cholesky = torch.linalg.cholesky(
                operator.diagonal_blocks - self.shift * eye
            )

    def __call__(self, residual, eigenvalues, davidson=False):
        if self.kind == "none":
            return residual
        if self.kind == "block":
            return torch.cholesky_solve(
                residual.reshape(self.op.n, self.op.b, -1), self.cholesky
            ).reshape(residual.shape)
        if davidson:
            denominator = self.op.diagonal[:, None] - eigenvalues[None, :]
            floor = max(1e-4, float(self.op.diagonal.abs().max()) * 1e-3)
            sign = torch.where(
                denominator < 0,
                -torch.ones_like(denominator),
                torch.ones_like(denominator),
            )
            return residual / (sign * denominator.abs().clamp_min(floor))
        return residual / (self.op.diagonal - self.shift)[:, None]


def orthogonalize(v, against=None):
    """Rank-revealing Gram orthogonalisation with reprojection and final QR."""
    if v.shape[1] == 0:
        return v
    if against is not None and against.shape[1]:
        for _ in range(2):
            v = v - against @ (against.T @ v)
    norms = v.norm(dim=0)
    keep = norms > 10 * torch.finfo(v.dtype).eps
    v = v[:, keep] / norms[keep]
    if v.shape[1] == 0:
        return v
    gram = v.T @ v
    values, vectors = torch.linalg.eigh((gram + gram.T) / 2)
    threshold = max(1e-10, 100 * torch.finfo(v.dtype).eps) * values[-1]
    keep = values > threshold
    v = (v @ vectors[:, keep]) / values[keep].sqrt()
    if against is not None and against.shape[1]:
        v = v - against @ (against.T @ v)
    capacity = v.shape[0] - (against.shape[1] if against is not None else 0)
    return torch.linalg.qr(v[:, :capacity], mode="reduced").Q


def initial_space(op, k, guess, seed):
    generator = torch.Generator(device=op.device).manual_seed(seed)
    x = (
        op.blocks.new_empty(op.size, 0)
        if guess is None
        else guess[:, :k].to(op.device, op.dtype)
    )
    if x.shape[1]:
        x = torch.linalg.qr(x, mode="reduced").Q
    if x.shape[1] < k:
        extra = torch.randn(
            op.size,
            k - x.shape[1],
            device=op.device,
            dtype=op.dtype,
            generator=generator,
        )
        # Householder QR completes a warm basis reliably even when k reaches
        # the full matrix dimension; Gram rank thresholds can drop the final
        # random complement direction in that case.
        x = torch.linalg.qr(torch.cat((x, extra), dim=1), mode="reduced").Q
    if x.shape[1] != k:
        raise RuntimeError("Could not construct requested initial subspace")
    return x


def ritz(v, av, k):
    projected = v.T @ av
    values, rotation = torch.linalg.eigh((projected + projected.T) / 2)
    rotation = rotation[:, :k]
    return values[:k], v @ rotation, av @ rotation, rotation


def davidson(
    op,
    k,
    guess=None,
    *,
    tolerance=1e-7,
    maxiter=100,
    preconditioner="jacobi",
    subspace_factor=3,
    seed=123,
):
    """Thick-restarted block Davidson with cached H times the search basis."""
    precondition = Preconditioner(op, preconditioner)
    v = initial_space(op, k, guess, seed)
    av = op(v)
    capacity = min(op.size, max(k + 1, subspace_factor * k))
    status = "maxiter"
    for iteration in range(1, maxiter + 1):
        values, x, ax, _ = ritz(v, av, k)
        residual = ax - x * values
        norms = residual.norm(dim=0)
        if float(norms.max()) <= tolerance:
            status = "converged"
            break
        active = norms > tolerance
        directions = precondition(residual[:, active], values[active], davidson=True)
        if v.shape[1] + directions.shape[1] > capacity:
            v, av = x, ax
        w = orthogonalize(directions, v)
        w = w[:, : capacity - v.shape[1]]
        if w.shape[1] == 0:
            status = "stagnated"
            break
        v, av = torch.cat((v, w), dim=1), torch.cat((av, op(w)), dim=1)
    return (
        values,
        x,
        dict(status=status, iterations=iteration, residual_max=float(norms.max())),
    )


def lobpcg(
    op,
    k,
    guess=None,
    *,
    tolerance=1e-7,
    maxiter=100,
    preconditioner="jacobi",
    subspace_factor=3,
    seed=123,
):
    """Operator-based block locally optimal preconditioned conjugate gradients.

    Each Rayleigh-Ritz space spans current X, preconditioned residual W and
    previous search directions P. Rank-deficient directions are removed.
    This is a benchmark implementation, not a wrapper around torch.lobpcg.
    """
    precondition = Preconditioner(op, preconditioner)
    x = initial_space(op, k, guess, seed)
    values, x, ax, _ = ritz(x, op(x), k)
    p = x[:, :0]
    status = "maxiter"
    for iteration in range(1, maxiter + 1):
        residual = ax - x * values
        norms = residual.norm(dim=0)
        if float(norms.max()) <= tolerance:
            status = "converged"
            break
        active = norms > tolerance
        p = orthogonalize(p, x)
        w = orthogonalize(
            precondition(residual[:, active], values[active]), torch.cat((x, p), dim=1)
        )
        if w.shape[1] + p.shape[1] == 0:
            status = "stagnated"
            break
        v = torch.cat((x, w, p), dim=1)
        av = torch.cat((ax, op(w), op(p)), dim=1)
        values, x, ax, rotation = ritz(v, av, k)
        p = v[:, k:] @ rotation[k:]
    # Recompute the final residual: the last iteration may have updated X.
    norms = (ax - x * values).norm(dim=0)
    if float(norms.max()) <= tolerance:
        status = "converged"
    return (
        values,
        x,
        dict(status=status, iterations=iteration, residual_max=float(norms.max())),
    )


def occupations(values, counts, tau, full_size):
    fillings = torch.stack([_fill(values, count, tau)[0] for count in counts], dim=1)
    # Conservative bound: all omitted eigenvalues are >= the last retained one.
    tail = float((full_size - len(values)) * fillings[-1].sum())
    return fillings, tail


def solve_occupied(op, method, counts, tau, empty_states, guess, args):
    k = min(
        op.size,
        max(
            1,
            math.ceil(float(counts.max())) + empty_states,
            guess.shape[1] if guess is not None else 0,
        ),
    )
    iterations = attempts = 0
    eigensolver_ms = occupation_ms = 0.0
    while True:
        (values, vectors, info), elapsed = timed(
            op.device,
            lambda: {"davidson": davidson, "lobpcg": lobpcg, "chebyshev": chebyshev}[
                method
            ](
                op,
                k,
                guess,
                tolerance=args.tolerance,
                maxiter=args.maxiter,
                preconditioner=args.preconditioner,
                subspace_factor=args.subspace_factor,
                seed=args.seed,
                **(
                    dict(degree=args.chebyshev_degree, guard=args.chebyshev_guard)
                    if method == "chebyshev"
                    else {}
                ),
            ),
        )
        eigensolver_ms += elapsed
        iterations += info["iterations"]
        attempts += 1
        (f, tail), elapsed = timed(
            op.device, lambda: occupations(values, counts, tau, op.size)
        )
        occupation_ms += elapsed
        if (
            info["status"] != "converged"
            or tail <= args.tail_tolerance
            or not args.auto_expand
            or k == op.size
        ):
            break
        k = min(op.size, k + max(empty_states, k // 4, 8))
        guess = vectors
    info.update(
        k=k,
        retained_fraction=k / op.size,
        iterations=iterations,
        attempts=attempts,
        tail_bound=tail,
        occupation_complete=tail <= args.tail_tolerance,
        eigensolver_ms=eigensolver_ms,
        occupation_ms=occupation_ms,
    )
    return values, vectors, f, info


def dense_warm_seed(values, vectors, counts, tau, args):
    """Select a population-complete starting space from the first dense solve."""
    k = min(len(values), max(1, math.ceil(float(counts.max())) + args.empty_states))
    while True:
        f, tail = occupations(values[:k], counts, tau, len(values))
        if tail <= args.tail_tolerance or k == len(values):
            return values[:k].clone(), vectors[:, :k].clone(), f, tail
        k = min(len(values), k + max(args.empty_states, k // 4, 8))


def repeat_configuration(atoms, repeat):
    multiplier = math.prod(repeat)
    result = atoms.repeat(repeat) if multiplier != 1 else atoms.copy()
    result.info = deepcopy(atoms.info)
    for key in ("N_alpha", "N_beta", "total_charge"):
        if key not in atoms.info:
            raise ValueError(f"Missing required atoms.info[{key!r}]")
        result.info[key] = float(atoms.info[key]) * multiplier
    if "elec_temp" not in atoms.info:
        raise ValueError('Missing required atoms.info["elec_temp"]')
    return result


def make_batch(calculator, atoms):
    config = config_from_atoms(
        atoms, key_specification=calculator.keyspec, head_name=calculator.head
    )
    data = ExtAtomicData.from_config(
        config,
        calculator.z_table,
        float(calculator.model.r_max),
        heads=calculator.model.heads,
        atomic_multipoles_max_l=1,
        preserve_cell=True,
    )
    return (
        next(iter(torch_geometric.dataloader.DataLoader([data], batch_size=1)))
        .to(calculator.device)
        .to_dict()
    )


def onsite_density(vectors, f, block_size):
    local = vectors.reshape(-1, block_size, vectors.shape[1])
    return torch.einsum("nak,k,nbk->nab", local, f.sum(1), local)


def validate(values, vectors, f, reference, block_size, counts, tolerance):
    """Untimed comparison of invariant observables, not individual eigenvectors."""
    orthogonality = float(
        (
            vectors.T @ vectors
            - torch.eye(len(values), device=vectors.device, dtype=vectors.dtype)
        )
        .abs()
        .max()
    )
    result = dict(
        orthogonality_max=orthogonality,
        count_error=float((f.sum(0) - counts).abs().max()),
    )
    if reference is not None:
        e, gamma, band, full_f = reference[:4]
        gamma_approx = onsite_density(vectors, f, block_size).cpu()
        result.update(
            eigenvalue_max_error=float((values.cpu() - e[: len(values)]).abs().max()),
            onsite_density_max_error=float((gamma_approx - gamma).abs().max()),
            band_energy_error_eV=float((values[:, None] * f).sum().cpu() - band),
            reference_omitted_electrons=float(full_f[len(values) :].sum()),
        )
        result["lowest_verified"] = result["eigenvalue_max_error"] <= max(
            10 * tolerance,
            100 * torch.finfo(values.dtype).eps * max(1.0, float(e.abs().max())),
        )
    else:
        result["lowest_verified"] = None
    return result


def parse_repeat(value):
    try:
        fields = tuple(int(x) for x in value.lower().replace("x", ",").split(","))
        if len(fields) == 1:
            fields = fields * 3
        if len(fields) != 3 or min(fields) < 1:
            raise ValueError
        return fields
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use a positive repetition such as 1,1,1 or 2,2,2"
        ) from exc


def method_label(row):
    label = row["method"]
    if row.get("capped_run"):
        label += f"[cap={row['iteration_cap']}]"
    return label


def plot_timings(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    groups = {}
    for row in rows:
        if row.get("status") != "converged" or not row.get("occupation_complete", True):
            continue
        if row.get("lowest_verified") is False:
            continue
        if row["start"] == "warm" and not row["warm_used"]:
            continue
        key = (method_label(row), row["backend"], row["start"])
        groups.setdefault(key, {}).setdefault(row["atoms"], []).append(
            row["occupied_space_ms"]
        )
    for key, points in groups.items():
        sizes = sorted(points)
        ax.loglog(
            sizes, [np.median(points[n]) for n in sizes], "-o", label="/".join(key)
        )
    ax.set(
        xlabel="Atoms",
        ylabel="Occupied-space time (ms; median across repeats/frames)",
        title="Occupied-space solves: converged, occupation-complete results only",
    )
    ax.grid(alpha=0.2)
    if groups:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "timings.png", dpi=180)
    plt.close(fig)


def plot_diagnostics(rows, output, tolerance=1e-7, tail_tolerance=1e-6):
    """Show every timed row, including failures and first-frame warm requests.

    Historical files store only the last repetition's outcome; their timing
    ranges must not be interpreted as repeated successful solves.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    groups = {}
    for row in rows:
        if "occupied_space_ms" in row:
            groups.setdefault(
                (method_label(row), row["backend"], row["start"]), []
            ).append(row)
    colors = {
        "dense": "black",
        "davidson": "tab:blue",
        "lobpcg": "tab:orange",
        "chebyshev": "tab:green",
        "foe": "tab:purple",
    }
    for (method, backend, start), group in groups.items():
        group.sort(key=lambda r: (r["atoms"], r["frame"]))
        x = [r["atoms"] for r in group]
        color = colors[method.split("[")[0]]
        style = "-" if start == "cold" else ":"
        label = f"{method}/{backend}/{start}"
        if start == "warm" and not any(r["warm_used"] for r in group):
            label = f"{method}/{backend}/warm requested (actually cold)"
        y = [r["occupied_space_ms"] / 1000 for r in group]
        axes[0, 0].plot(x, y, style, color=color, alpha=0.65, label=label)
        for r, seconds in zip(group, y):
            accepted = (
                r["status"] == "converged"
                and r.get("occupation_complete", True)
                and r.get("lowest_verified") is not False
            )
            # A legacy successful final run with a much shorter median is ambiguous.
            ambiguous = (
                "sample_results" not in r
                and r["method"] != "dense"
                and accepted
                and max(r.get("solve_times_ms", [1]))
                > 1.5 * min(r.get("solve_times_ms", [1]))
            )
            marker = "D" if ambiguous else ("o" if accepted else "x")
            axes[0, 0].scatter(r["atoms"], seconds, marker=marker, color=color, s=55)
            samples = r.get("solve_times_ms", [])
            if samples:
                offset = r.get("occupation_ms", 0) if method == "dense" else 0
                axes[0, 0].vlines(
                    r["atoms"],
                    (min(samples) + offset) / 1000,
                    (max(samples) + offset) / 1000,
                    color=color,
                    alpha=0.45,
                )
        if method == "foe":
            continue  # No eigenpair residual or retained eigenspace for FOE.
        axes[0, 1].plot(
            x, [r["k"] / r["dimension"] for r in group], style + "o", color=color
        )
        axes[1, 0].plot(
            x,
            [max(r.get("residual_max", 0), 1e-17) for r in group],
            style + "o",
            color=color,
        )
        axes[1, 1].plot(
            x,
            [max(r.get("reference_omitted_electrons", 0), 1e-17) for r in group],
            style + "o",
            color=color,
        )
    axes[0, 0].set(
        xscale="log",
        yscale="log",
        ylabel="Occupied-space time (s)",
        title="All runs; bars show timing range",
    )
    axes[0, 1].set(
        xscale="log",
        ylabel="Retained states / full dimension",
        ylim=(0, 1.05),
        title="Final repetition: retained fraction",
    )
    axes[1, 0].set(
        xscale="log",
        yscale="log",
        ylabel="Maximum eigenpair residual (eV)",
        title="Final repetition: residual",
    )
    axes[1, 1].set(
        xscale="log",
        yscale="log",
        ylabel="Omitted electrons from dense reference",
        title="Final repetition: occupation truncation",
    )
    axes[1, 0].axhline(tolerance, color="0.5", linestyle="--")
    axes[1, 1].axhline(tail_tolerance, color="0.5", linestyle="--")
    for ax in axes.flat:
        ax.set_xlabel("Atoms")
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [
        Line2D([], [], color="0.3", marker="o", linestyle="", label="Accepted outcome"),
        Line2D(
            [],
            [],
            color="0.3",
            marker="x",
            linestyle="",
            label="Unconverged/incomplete",
        ),
        Line2D(
            [],
            [],
            color="0.3",
            marker="D",
            linestyle="",
            label="Ambiguous historical repeats",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8)
    fig.suptitle(
        "Sparse solver diagnostics — failed runs are not equivalent-accuracy timings"
    )
    fig.tight_layout(rect=(0, 0.18, 1, 0.95))
    fig.savefig(output / "diagnostics.png", dpi=180)
    plt.close(fig)


def plot_trajectory(rows, output):
    """Per-frame costs, separated by system size; dense seeds remain visible."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = sorted({tuple(r["repeat"]) for r in rows if "occupied_space_ms" in r})
    if not cases:
        return
    fig, axes = plt.subplots(len(cases), 1, figsize=(10, 4 * len(cases)), squeeze=False)
    for ax, case in zip(axes[:, 0], cases):
        groups = {}
        for row in rows:
            if tuple(row["repeat"]) == case and "occupied_space_ms" in row:
                groups.setdefault(
                    (method_label(row), row["backend"], row["start"]), []
                ).append(row)
        for key, group in groups.items():
            group.sort(key=lambda r: r["frame"])
            (line,) = ax.plot(
                [r["frame"] for r in group],
                [r["occupied_space_ms"] / 1000 for r in group],
                "--" if key[2] == "warm" else "-",
                label="/".join(key),
            )
            for r in group:
                ok = (
                    r["status"] == "converged"
                    and r.get("occupation_complete", True)
                    and r.get("lowest_verified") is not False
                )
                marker = (
                    "s" if r.get("initialization") == "dense" else ("o" if ok else "x")
                )
                # Hollow circles denote a requested warm run that actually fell back to cold.
                hollow = key[2] == "warm" and not r.get("warm_used") and marker == "o"
                ax.plot(
                    r["frame"],
                    r["occupied_space_ms"] / 1000,
                    marker=marker,
                    color=line.get_color(),
                    markerfacecolor="none" if hollow else line.get_color(),
                )
        ax.set(
            xlabel="Trajectory frame (selected sequence, zero-based)",
            ylabel="Solve + occupations (s)",
            yscale="log",
            title=f"Replication {case}",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Frame timings: square = dense seed; cross = unresolved; hollow = cold fallback"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output / "trajectory_timings.png", dpi=180)
    plt.close(fig)


def plot_observables(rows, output):
    """Compare density-producing costs and errors; keep unresolved runs visible."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    groups = {}
    for row in rows:
        if "electronic_observables_ms" in row:
            groups.setdefault(
                (method_label(row), row["backend"], row["start"]), []
            ).append(row)
    fields = [
        "electronic_observables_ms",
        "onsite_density_max_error",
        "band_energy_error_eV",
    ]
    labels = [
        "Solve + onsite density (ms)",
        "Maximum onsite density error",
        "Absolute band energy error (eV)",
    ]
    for key, group in groups.items():
        group.sort(key=lambda r: (r["atoms"], r["frame"]))
        for ax, field, label in zip(axes, fields, labels):
            valid = [r for r in group if field in r]
            (line,) = ax.plot(
                [r["atoms"] for r in valid],
                [max(abs(r[field]), 1e-17) for r in valid],
                label="/".join(key),
            )
            for r in valid:
                accepted = (
                    r["status"] == "converged"
                    and r.get("occupation_complete", True)
                    and r.get("lowest_verified") is not False
                )
                ax.scatter(
                    r["atoms"],
                    max(abs(r[field]), 1e-17),
                    marker="o" if accepted else "x",
                    color=line.get_color(),
                )
    for ax, label in zip(axes, labels):
        ax.set(xscale="log", yscale="log", xlabel="Atoms", ylabel=label)
        ax.grid(alpha=0.2)
    if groups:
        axes[0].legend(fontsize=7)
    fig.suptitle("Electronic observables: crosses mark unresolved/incomplete results")
    fig.tight_layout()
    fig.savefig(output / "observables.png", dpi=180)
    plt.close(fig)


def benchmark_foe(op, counts, tau, args, base, reference, setup_ms, record):
    """FOE has no trajectory guess; every timed sample rebuilds its moments."""

    def run():
        return polynomial_density(
            op,
            counts,
            tau,
            degree=args.foe_degree,
            chunk_size=args.foe_chunk_size,
            tolerance=args.foe_tolerance,
            count_tolerance=args.tail_tolerance,
        )

    row = dict(
        base,
        method="foe",
        backend=op.backend,
        start="cold",
        warm_used=False,
        operator_setup_ms=setup_ms,
    )
    try:
        for _ in range(args.warmup):
            run()
        baseline = (
            torch.cuda.memory_allocated(op.device) if op.device.type == "cuda" else 0
        )
        if op.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(op.device)
        samples, times = [], []
        for _ in range(args.repeats):
            op.calls = op.vector_products = 0
            (gamma, band, info), elapsed = timed(op.device, run)
            samples.append(
                dict(
                    info,
                    occupied_space_ms=elapsed,
                    matvec_calls=op.calls,
                    vector_products=op.vector_products,
                )
            )
            times.append(elapsed)
        peak = (
            torch.cuda.max_memory_allocated(op.device)
            if op.device.type == "cuda"
            else 0
        )
        row.update(info)
        row["electronic_observables_ms"] = float(np.median(times))
        if reference is not None and len(reference) > 4:
            row["free_band_energy_error_eV"] = info["free_band_energy_eV"] - float(
                reference[4]
            )
        row.update(
            sample_results=samples,
            solve_times_ms=times,
            occupied_space_ms=float(np.median(times)),
            solve_ms=float(np.median(times)),
            matvec_calls=op.calls,
            vector_products=op.vector_products,
            successful_repeats=sum(s["status"] == "converged" for s in samples),
            peak_cuda_MB=peak / 1e6 if peak else None,
            incremental_cuda_MB=(peak - baseline) / 1e6 if peak else None,
        )
        if len({s["status"] for s in samples}) != 1:
            row["status"] = "mixed_repeats"
        row["occupation_complete"] = all(s["occupation_complete"] for s in samples)
        if reference is not None:
            row.update(
                onsite_density_max_error=float(
                    (gamma.cpu() - reference[1]).abs().max()
                ),
                band_energy_error_eV=float(band.cpu() - reference[2]),
            )
    except RuntimeError as error:
        row.update(status="failed", error=str(error))
        if op.device.type == "cuda":
            torch.cuda.empty_cache()
    record(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--xyz", required=True)
    parser.add_argument(
        "--frames", default=":", help="ASE frame slice; default all, in file order"
    )
    parser.add_argument("--output", type=Path, default=Path("occupied_space_profile"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--head")
    parser.add_argument(
        "--repeat",
        action="append",
        type=parse_repeat,
        help="Repeatable; default 1,1,1 / 2,1,1 / 2,2,1 / 2,2,2",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["dense", "davidson", "lobpcg", "chebyshev", "foe"],
        default=["dense", "davidson", "lobpcg"],
    )
    parser.add_argument(
        "--backends", nargs="+", choices=["csr", "edges"], default=["csr"]
    )
    parser.add_argument(
        "--starts", nargs="+", choices=["cold", "warm"], default=["cold", "warm"]
    )
    parser.add_argument("--chebyshev-degree", type=int, default=20)
    parser.add_argument("--chebyshev-guard", type=int, default=16)
    parser.add_argument("--foe-degree", type=int, default=256)
    parser.add_argument("--foe-chunk-size", type=int, default=128)
    parser.add_argument(
        "--foe-tolerance",
        type=float,
        default=1e-8,
        help="Sampled scalar Fermi-function approximation tolerance",
    )
    parser.add_argument("--empty-states", type=int, default=16)
    parser.add_argument(
        "--tail-tolerance",
        type=float,
        default=1e-6,
        help="Bound on total omitted alpha+beta population",
    )
    parser.add_argument("--no-auto-expand", dest="auto_expand", action="store_false")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-7,
        help="Absolute eigenpair residual tolerance in eV",
    )
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument(
        "--iteration-caps",
        nargs="+",
        type=int,
        help="Compare fixed iteration budgets; keep capped warm guesses and disable space expansion",
    )
    parser.add_argument(
        "--subspace-factor",
        type=int,
        default=3,
        help="Davidson restart space / requested states",
    )
    parser.add_argument(
        "--preconditioner", choices=["none", "jacobi", "block"], default="jacobi"
    )
    parser.add_argument("--edge-chunk", type=int, default=1024)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Timed repetitions, always using the same preceding-frame guess",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Untimed solver repetitions per frame/method/start",
    )
    parser.add_argument(
        "--dense-max-dim",
        type=int,
        default=8192,
        help="Skip dense reference above this dimension",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    if (
        min(
            args.chebyshev_degree,
            args.chebyshev_guard,
            args.foe_degree,
            args.foe_chunk_size,
            args.maxiter,
            args.repeats,
            args.edge_chunk,
            args.threads,
            args.dense_max_dim,
        )
        < 1
        or args.empty_states < 1
        or args.warmup < 0
        or args.subspace_factor < 2
        or args.foe_tolerance <= 0
        or args.tolerance <= 0
        or args.tail_tolerance <= 0
    ):
        parser.error(
            "Require positive limits/tolerances, empty-states >=1, subspace-factor >=2 and warmup >=0"
        )
    if args.iteration_caps and (
        min(args.iteration_caps) < 1 or "dense" not in args.methods
    ):
        parser.error(
            "--iteration-caps requires positive caps and dense in --methods for error measurement"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable; use --device cpu for a correctness smoke test"
        )
    if not (args.device.startswith("cuda") or args.device == "cpu"):
        parser.error("This benchmark supports CUDA and CPU")
    if args.dtype == "float32" and args.tolerance < 1e-5:
        warnings.warn("float32 may not reach this tolerance; consider --tolerance 1e-4")
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(getattr(torch, args.dtype))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    calculator = MLDFTBCalculator(
        model_path=args.model,
        device=args.device,
        default_dtype=args.dtype,
        head=args.head,
    )
    device = calculator.device
    frames = read(args.xyz, index=args.frames)
    if not isinstance(frames, list):
        frames = [frames]
    if not frames:
        parser.error("No input frames selected")
    for frame in frames[1:]:
        if not np.array_equal(frame.numbers, frames[0].numbers):
            parser.error(
                "Trajectory warm starts require the same atom count and ordering/species"
            )
    repetitions = args.repeat or [(1, 1, 1), (2, 1, 1), (2, 2, 1), (2, 2, 2)]
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    settings.update(
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        hamiltonian_cutoff=float(calculator.model.hamiltonian.r_max),
        mace_cutoff=float(calculator.model.r_max),
        block_size=calculator.model.hamiltonian.block_size,
        selected_frames=len(frames),
        actual_repetitions=repetitions,
    )
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    rows = []

    def record(row):
        rows.append(row)
        with (args.output / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    # Fresh run: do not mix timing records from previous invocations.
    (args.output / "results.jsonl").write_text("")
    with torch.no_grad():
        for repeat in repetitions:
            warm_cache = {}
            for frame_index, original in enumerate(frames):
                atoms = repeat_configuration(original, repeat)
                data, graph_ms = timed(device, lambda: make_batch(calculator, atoms))
                features, feature_ms = timed(
                    device, lambda: node_features(calculator.model, data)
                )
                (blocks, edges, onsite), block_ms = timed(
                    device,
                    lambda: hamiltonian_blocks(
                        calculator.model.hamiltonian,
                        features,
                        data["positions"],
                        data["edge_index"],
                        data["shifts"],
                    ),
                )
                op, coalesce_ms = timed(
                    device,
                    lambda: BlockHamiltonian(blocks, edges, onsite, args.edge_chunk),
                )
                counts = torch.tensor(
                    [atoms.info["N_alpha"], atoms.info["N_beta"]], device=device
                )
                tau = calculator.model.electronic_state.occupations.thermal_energy(
                    torch.tensor(atoms.info["elec_temp"], device=device)
                )
                if (
                    not bool(torch.isfinite(counts).all())
                    or bool((counts < 0).any())
                    or float(counts.max()) > op.size
                ):
                    raise ValueError("Invalid spin count for replicated basis")
                if not bool(torch.isfinite(tau)) or float(tau) < 0:
                    raise ValueError("Smearing must be finite and nonnegative")
                nuclei = calculator.model.density_electrostatics.nuclei
                if nuclei.charge_mode == "per_species":
                    nuclear_charge = (
                        data["node_attrs"] @ nuclei.effective_charges
                    ).sum()
                else:
                    nuclear_charge = data["effective_nuclear_charges"].sum()
                if (
                    abs(
                        float(nuclear_charge - counts.sum())
                        - float(atoms.info["total_charge"])
                    )
                    > 1e-5
                ):
                    raise ValueError(
                        "Input electron counts and total charge disagree with model effective nuclei"
                    )
                base = dict(
                    frame=frame_index,
                    repeat=list(repeat),
                    atoms=len(atoms),
                    dimension=op.size,
                    mace_edges=data["edge_index"].shape[1],
                    hamiltonian_edges=edges.shape[1],
                    stored_blocks=len(op.blocks),
                    scalar_nnz=len(op.blocks) * op.b**2,
                    block_density=len(op.blocks) / op.n**2,
                    alpha=float(counts[0]),
                    beta=float(counts[1]),
                    tau_eV=float(tau),
                    graph_ms=graph_ms,
                    feature_ms=feature_ms,
                    block_ms=block_ms,
                    coalesce_ms=coalesce_ms,
                )
                del data, features, blocks, edges, onsite
                reference = None
                seed_vectors = seed_row = None
                need_seed = (
                    frame_index == 0
                    and "warm" in args.starts
                    and any(m not in ("dense", "foe") for m in args.methods)
                )
                if (need_seed or args.iteration_caps) and op.size > args.dense_max_dim:
                    raise ValueError(
                        "Dense warm initialization exceeds --dense-max-dim; increase that limit or use --starts cold"
                    )
                if (
                    "dense" in args.methods or need_seed
                ) and op.size <= args.dense_max_dim:
                    try:
                        H, assembly_ms = timed(device, op.dense)
                        for _ in range(args.warmup):
                            torch.linalg.eigh(H)
                        times = []
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)
                        baseline = (
                            torch.cuda.memory_allocated(device)
                            if device.type == "cuda"
                            else 0
                        )
                        for sample in range(args.repeats):
                            e = u = (
                                None  # Release preceding timing outputs before allocation.
                            )
                            (e, u), milliseconds = timed(
                                device, lambda: torch.linalg.eigh(H)
                            )
                            times.append(milliseconds)
                        peak = (
                            torch.cuda.max_memory_allocated(device)
                            if device.type == "cuda"
                            else 0
                        )
                        (f, _), occupation_ms = timed(
                            device, lambda: occupations(e, counts, tau, op.size)
                        )
                        gamma, density_ms = timed(
                            device, lambda: onsite_density(u, f, op.b)
                        )
                        entropy_term = (
                            torch.special.xlogy(f, f)
                            + torch.special.xlogy(1 - f, 1 - f)
                        ).sum()
                        free_band = (e[:, None] * f).sum() + tau * entropy_term
                        reference = (
                            e.cpu(),
                            gamma.cpu(),
                            (e[:, None] * f).sum().cpu(),
                            f.cpu(),
                            free_band.cpu(),
                        )
                        dense_row = dict(
                            base,
                            method="dense",
                            backend="dense",
                            start="cold",
                            warm_used=False,
                            status="converged",
                            occupation_complete=True,
                            k=op.size,
                            retained_fraction=1.0,
                            iterations=1,
                            assembly_ms=assembly_ms,
                            density_ms=density_ms,
                            electronic_observables_ms=float(np.median(times))
                            + occupation_ms
                            + density_ms,
                            free_band_energy_eV=float(free_band),
                            occupation_ms=occupation_ms,
                            eigensolver_ms=float(np.median(times)),
                            occupied_space_ms=float(np.median(times)) + occupation_ms,
                            solve_ms=float(np.median(times)),
                            solve_times_ms=times,
                            residual_max=float((H @ u - u * e).norm(dim=0).max()),
                            peak_cuda_MB=(
                                peak / 1e6 if device.type == "cuda" else None
                            ),
                            incremental_cuda_MB=(
                                (peak - baseline) / 1e6
                                if device.type == "cuda"
                                else None
                            ),
                            **validate(
                                e, u, f, reference, op.b, counts, args.tolerance
                            ),
                        )
                        if "dense" in args.methods:
                            record(dense_row)
                        if need_seed:
                            (seed_values, seed_vectors, seed_f, seed_tail), seed_ms = (
                                timed(
                                    device,
                                    lambda: dense_warm_seed(e, u, counts, tau, args),
                                )
                            )
                            seed_row = dict(dense_row)
                            seed_row.update(
                                initialization="dense",
                                start="warm",
                                warm_used=False,
                                seed_selection_ms=seed_ms,
                                tail_bound=seed_tail,
                                k=len(seed_values),
                                retained_fraction=len(seed_values) / op.size,
                                occupied_space_ms=dense_row["occupied_space_ms"]
                                + seed_ms,
                                electronic_observables_ms=dense_row[
                                    "electronic_observables_ms"
                                ]
                                + seed_ms,
                            )
                            seed_row.update(
                                validate(
                                    seed_values,
                                    seed_vectors,
                                    seed_f,
                                    reference,
                                    op.b,
                                    counts,
                                    args.tolerance,
                                )
                            )
                            del seed_values, seed_f
                        del H, e, u, f, gamma, free_band
                    except torch.cuda.OutOfMemoryError:
                        record(
                            dict(
                                base,
                                method="dense",
                                backend="dense",
                                start="cold",
                                warm_used=False,
                                status="out_of_memory",
                            )
                        )
                        H = e = u = f = None
                        torch.cuda.empty_cache()
                elif "dense" in args.methods:
                    record(
                        dict(
                            base,
                            method="dense",
                            backend="dense",
                            start="cold",
                            warm_used=False,
                            status="skipped_dimension_limit",
                        )
                    )
                iterative_backends = (
                    args.backends if any(m != "dense" for m in args.methods) else []
                )
                for backend in iterative_backends:
                    try:
                        _, setup_ms = timed(device, lambda: op.prepare(backend))
                    except RuntimeError as error:
                        record(
                            dict(
                                base,
                                method="operator_setup",
                                backend=backend,
                                start="cold",
                                warm_used=False,
                                status="failed",
                                error=str(error),
                            )
                        )
                        op.csr = None
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        continue
                    method_caps = [
                        (m, cap)
                        for m in args.methods
                        for cap in (args.iteration_caps or [args.maxiter])
                        if m not in ("dense", "foe")
                    ]
                    if "foe" in args.methods:
                        method_caps.append(("foe", args.maxiter))
                    for method, iteration_cap in method_caps:
                        solver_args = deepcopy(args)
                        solver_args.maxiter = iteration_cap
                        if args.iteration_caps:
                            solver_args.auto_expand = False
                        if method == "dense":
                            continue
                        if method == "foe":
                            benchmark_foe(
                                op, counts, tau, args, base, reference, setup_ms, record
                            )
                            continue
                        for start in args.starts:
                            key = (backend, method, start, iteration_cap)
                            if start == "warm" and frame_index == 0:
                                if seed_vectors is None:
                                    record(
                                        dict(
                                            base,
                                            method=method,
                                            backend=backend,
                                            start=start,
                                            warm_used=False,
                                            status="dense_initialization_failed",
                                        )
                                    )
                                else:
                                    warm_cache[key] = seed_vectors
                                    record(
                                        dict(
                                            seed_row,
                                            method=method,
                                            backend=backend,
                                            iteration_cap=iteration_cap,
                                            capped_run=bool(args.iteration_caps),
                                        )
                                    )
                                continue
                            guess = warm_cache.get(key) if start == "warm" else None
                            warm_used = guess is not None

                            def run():
                                return solve_occupied(
                                    op,
                                    method,
                                    counts,
                                    tau,
                                    args.empty_states,
                                    guess,
                                    solver_args,
                                )

                            try:
                                for _ in range(args.warmup):
                                    run()
                                times = []
                                sample_results = []
                                if device.type == "cuda":
                                    torch.cuda.reset_peak_memory_stats(device)
                                baseline = (
                                    torch.cuda.memory_allocated(device)
                                    if device.type == "cuda"
                                    else 0
                                )
                                for sample in range(args.repeats):
                                    e = u = f = None
                                    op.calls = op.vector_products = 0
                                    (e, u, f, info), milliseconds = timed(device, run)
                                    times.append(milliseconds)
                                    sample_results.append(
                                        dict(
                                            info,
                                            occupied_space_ms=milliseconds,
                                            matvec_calls=op.calls,
                                            vector_products=op.vector_products,
                                        )
                                    )
                                statuses = {
                                    sample["status"] for sample in sample_results
                                }
                                aggregate = dict(info)
                                aggregate.update(
                                    status=(
                                        info["status"]
                                        if len(statuses) == 1
                                        else "mixed_repeats"
                                    ),
                                    occupation_complete=all(
                                        sample["occupation_complete"]
                                        for sample in sample_results
                                    ),
                                    sample_results=sample_results,
                                    successful_repeats=sum(
                                        sample["status"] == "converged"
                                        and sample["occupation_complete"]
                                        for sample in sample_results
                                    ),
                                    k_min=min(sample["k"] for sample in sample_results),
                                    k_max=max(sample["k"] for sample in sample_results),
                                )
                                peak = (
                                    torch.cuda.max_memory_allocated(device)
                                    if device.type == "cuda"
                                    else 0
                                )
                                _, density_ms = timed(
                                    device, lambda: onsite_density(u, f, op.b)
                                )
                                checks = validate(
                                    e, u, f, reference, op.b, counts, args.tolerance
                                )
                                converged_guess = (
                                    info["status"] == "converged"
                                    and info["occupation_complete"]
                                    and checks["lowest_verified"] is not False
                                )
                                reuse_guess = start == "warm" and (
                                    converged_guess
                                    or (
                                        bool(args.iteration_caps)
                                        and bool(torch.isfinite(u).all())
                                        and bool(torch.isfinite(e).all())
                                    )
                                )
                                record(
                                    dict(
                                        base,
                                        method=method,
                                        backend=backend,
                                        start=start,
                                        warm_used=warm_used,
                                        iteration_cap=iteration_cap,
                                        capped_run=bool(args.iteration_caps),
                                        retained_unconverged_guess=reuse_guess
                                        and not converged_guess,
                                        operator_setup_ms=setup_ms,
                                        density_ms=density_ms,
                                        electronic_observables_ms=float(
                                            np.median(times)
                                        )
                                        + density_ms,
                                        occupied_space_ms=float(np.median(times)),
                                        solve_ms=float(np.median(times)),
                                        solve_times_ms=times,
                                        matvec_calls=op.calls,
                                        vector_products=op.vector_products,
                                        peak_cuda_MB=(
                                            peak / 1e6
                                            if device.type == "cuda"
                                            else None
                                        ),
                                        incremental_cuda_MB=(
                                            (peak - baseline) / 1e6
                                            if device.type == "cuda"
                                            else None
                                        ),
                                        **aggregate,
                                        **checks,
                                    )
                                )
                                if reuse_guess:
                                    warm_cache[key] = u.detach()
                                else:
                                    warm_cache.pop(key, None)
                                del e, u, f
                            except torch.cuda.OutOfMemoryError:
                                e = u = f = None
                                warm_cache.pop(key, None)
                                record(
                                    dict(
                                        base,
                                        method=method,
                                        backend=backend,
                                        start=start,
                                        warm_used=warm_used,
                                        status="out_of_memory",
                                    )
                                )
                                torch.cuda.empty_cache()
                            except RuntimeError as error:
                                e = u = f = None
                                warm_cache.pop(key, None)
                                record(
                                    dict(
                                        base,
                                        method=method,
                                        backend=backend,
                                        start=start,
                                        warm_used=warm_used,
                                        status="failed",
                                        error=str(error),
                                    )
                                )
                del op, reference, seed_vectors, seed_row
    fields = sorted({key for row in rows for key in row})
    with (args.output / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if not args.no_plot:
        plot_trajectory(rows, args.output)
        plot_observables(rows, args.output)
        plot_timings(rows, args.output)
        plot_diagnostics(rows, args.output, args.tolerance, args.tail_tolerance)


if __name__ == "__main__":
    main()
