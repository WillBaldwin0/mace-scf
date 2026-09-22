"""Cold-start polynomial benchmarks. No dense-reference information is used."""

import math
import torch


def spectral_interval(op):
    # Gershgorin bounds are safe, unlike unqualified Lanczos extremal estimates.
    radius = (
        op.blocks.new_zeros(op.n, op.b)
        .index_add_(0, op.rows, op.blocks.abs().sum(-1))
        .flatten()
        - op.diagonal.abs()
    )
    lo = (op.diagonal - radius).min()
    hi = (op.diagonal + radius).max()
    pad = (
        100
        * torch.finfo(op.dtype).eps
        * torch.maximum(torch.ones_like(lo), torch.maximum(lo.abs(), hi.abs()))
    )
    return lo - pad, hi + pad


def chebyshev(
    op,
    k,
    guess=None,
    *,
    tolerance=1e-7,
    maxiter=100,
    seed=123,
    degree=20,
    guard=16,
    **unused
):
    """Chebyshev-filtered subspace iteration with normalized recurrence."""
    from .occupied_space import initial_space, ritz

    lo, hi = spectral_interval(op)
    width = min(op.size, k + guard)
    x = initial_space(op, width, guess, seed)
    values, x, ax, _ = ritz(x, op(x), width)
    status = "maxiter"
    for iteration in range(1, maxiter + 1):
        norms = (ax[:, :k] - x[:, :k] * values[:k]).norm(dim=0)
        if float(norms.max()) <= tolerance:
            status = "converged"
            break
        # First guard eigenvalue estimates the bottom of the unwanted interval.
        cutoff = values[min(k, width - 1)]
        half = (hi - cutoff) / 2
        center = (hi + cutoff) / 2
        t0 = (lo - center) / half
        rho = 1 / t0
        previous = x
        current = rho * (ax - center * x) / half
        for _ in range(2, degree + 1):
            next_rho = 1 / (2 * t0 - rho)
            nxt = 2 * next_rho * (op(current) - center * current) / half
            nxt -= next_rho * rho * previous
            previous, current, rho = current, nxt, next_rho
        x = torch.linalg.qr(current, mode="reduced").Q
        values, x, ax, _ = ritz(x, op(x), width)
    norms = (ax[:, :k] - x[:, :k] * values[:k]).norm(dim=0)
    if float(norms.max()) <= tolerance:
        status = "converged"
    return (
        values[:k],
        x[:, :k],
        dict(
            status=status,
            iterations=iteration,
            residual_max=float(norms.max()),
            filter_degree=degree,
            filter_subspace=width,
            spectral_lower=float(lo),
            spectral_upper=float(hi),
        ),
    )


def polynomial_density(
    op, counts, tau, *, degree=256, chunk_size=128, tolerance=1e-8, count_tolerance=1e-6
):
    """Exact-column Chebyshev moments, followed by canonical Fermi expansion.

    Returns sum-spin onsite P blocks and Tr(HP). Cost is O(degree * nnz * M),
    not linear scaling. Workspace O(M * chunk + degree * N * B**2).
    Chemical-potential iterations and both spins reuse the same moments.
    Positive smearing is required; no eigenvectors or full density are formed.
    """
    if float(tau) <= 0:
        raise RuntimeError("Polynomial Fermi expansion requires positive elec_temp")
    lo, hi = spectral_interval(op)
    center, half = (hi + lo) / 2, (hi - lo) / 2
    moments = op.blocks.new_zeros(degree + 2, op.n, op.b, op.b)
    # Atom-aligned column chunks allow direct extraction of all onsite blocks.
    atoms_per_chunk = max(1, chunk_size // op.b)
    local = torch.arange(op.b, device=op.device)
    for first in range(0, op.n, atoms_per_chunk):
        last = min(op.n, first + atoms_per_chunk)
        cols = torch.arange(first * op.b, last * op.b, device=op.device)
        columns = torch.arange(len(cols), device=op.device)
        previous = op.blocks.new_zeros(op.size, len(cols))
        previous[cols, columns] = 1
        rows = torch.arange(first, last, device=op.device)[:, None] * op.b + local
        outcols = torch.arange(last - first, device=op.device)[:, None] * op.b + local

        def save(j, value):
            moments[j, first:last] = value[rows[:, :, None], outcols[:, None, :]]

        save(0, previous)
        current = (op(previous) - center * previous) / half
        save(1, current)
        for j in range(2, degree + 2):
            nxt = 2 * (op(current) - center * current) / half - previous
            save(j, nxt)
            previous, current = current, nxt
    trace = moments.diagonal(dim1=-2, dim2=-1).sum((-1, -2))
    # Oversampled Gauss-Chebyshev quadrature; only small coefficient arrays.
    q = 4 * (degree + 1)
    theta = (torch.arange(q, device=op.device, dtype=op.dtype) + 0.5) * math.pi / q
    orders = torch.arange(degree + 1, device=op.device, dtype=op.dtype)
    cosine = torch.cos(orders[:, None] * theta)
    energy = center + half * torch.cos(theta)
    weights = op.blocks.new_full((degree + 1,), 2 / q)
    weights[0] /= 2
    transform = weights[:, None] * cosine
    # Contract trace before mu search: each iteration is O(degree), not O(M).
    trace_weights = trace[: degree + 1] @ transform
    coefficients, mus, errors, count_errors = [], [], [], []
    free_band = op.blocks.new_zeros(())
    for spin, count in enumerate(counts):
        if spin == 1 and float(count) == float(counts[0]):
            coefficients.append(coefficients[0])
            mus.append(mus[0])
            errors.append(errors[0])
            count_errors.append(count_errors[0])
            free_band = 2 * free_band
            continue
        number = float(count)
        if number == 0 or number == op.size:
            c = torch.zeros_like(orders)
            c[0] = number / op.size
            mu = lo - 100 * tau if number == 0 else hi + 100 * tau
            error = 0.0
        else:
            lower, upper = lo - 100 * tau, hi + 100 * tau
            for _ in range(80):
                mu = (lower + upper) / 2
                population = trace_weights @ torch.sigmoid((mu - energy) / tau)
                lower = torch.where(population < count, mu, lower)
                upper = torch.where(population < count, upper, mu)
            c = transform @ torch.sigmoid((mu - energy) / tau)
            # Independent uniform grid, including endpoints, via Clenshaw.
            grid = torch.linspace(
                -1, 1, 8 * (degree + 1), device=op.device, dtype=op.dtype
            )
            b1, b2 = torch.zeros_like(grid), torch.zeros_like(grid)
            for j in range(degree, 0, -1):
                b0 = 2 * grid * b1 - b2 + c[j]
                b2, b1 = b1, b0
            approximation = grid * b1 - b2 + c[0]
            error = float(
                (approximation - torch.sigmoid((mu - center - half * grid) / tau))
                .abs()
                .max()
            )
        if number == 0:
            pass
        elif number == op.size:
            free_band += center * trace[0] + half * trace[1]
        else:
            free_values = mu * torch.sigmoid(
                (mu - energy) / tau
            ) - tau * torch.nn.functional.softplus((mu - energy) / tau)
            free_band += trace_weights @ free_values
        coefficients.append(c)
        mus.append(float(mu))
        errors.append(error)
        count_errors.append(float((trace[: degree + 1] @ c - count).abs()))
    c = coefficients[0] + coefficients[1]
    gamma = torch.einsum("j,jnab->nab", c, moments[: degree + 1])
    # x*T_0=T_1; x*T_j=(T_{j-1}+T_{j+1})/2 for j>=1.
    htrace = center * trace[: degree + 1]
    htrace = htrace.clone()
    htrace[0] += half * trace[1]
    htrace[1:] += half * (trace[:degree] + trace[2 : degree + 2]) / 2
    band = c @ htrace
    accepted = max(errors) <= tolerance and max(count_errors) <= count_tolerance
    return (
        gamma,
        band,
        dict(
            status="converged" if accepted else "polynomial_unresolved",
            occupation_complete=accepted,
            polynomial_degree=degree,
            polynomial_grid_error=max(errors),
            count_error=max(count_errors),
            chemical_potentials=mus,
            free_band_energy_eV=float(free_band),
            spectral_lower=float(lo),
            spectral_upper=float(hi),
            iterations=degree,
            k=0,
            retained_fraction=0.0,
            # Acceptance is sampled scalar-function accuracy, not an eigenpair test.
            acceptance="sampled_fermi_function_and_particle_count",
        ),
    )
