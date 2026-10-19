"""Explicit fixed-population spectral response and double backward.

No eigenvector derivatives or matrix-function reconstruction. Cubic intermediates
are permitted; optional row tiling bounds storage without changing the formula.
"""

from functools import lru_cache

import torch


def _derivatives(x, mu, tau, saturated=False):
    f = torch.sigmoid((mu - x) / tau)
    p = f * (1 - f)
    if saturated:
        p = torch.zeros_like(p)
    return (
        -p / tau,
        p * (1 - 2 * f) / tau**2,
        -p * (1 - 6 * f + 6 * f * f) / tau**3,
        p * (1 - 14 * f + 36 * f * f - 24 * f**3) / tau**4,
    )


def first_divided(eps, f, mu, tau, saturated=False):
    gap = eps[:, None] - eps[None, :]
    near = gap.abs() <= tau * 1e-3
    mid = (eps[:, None] + eps[None, :]) / 2
    fp, _, f3, _ = _derivatives(mid, mu, tau, saturated)
    local = fp + gap.square() * f3 / 24
    return torch.where(
        near,
        local,
        (f[:, None] - f[None, :]) / torch.where(near, torch.ones_like(gap), gap),
    )


def second_divided(eps, mu, tau, divided, start=0, stop=None, saturated=False):
    """Coefficients [i,k,j], sorting arguments to divide by the largest gap."""
    rows = torch.arange(start, eps.numel() if stop is None else stop, device=eps.device)
    return _second_divided_rows(eps, mu, tau, divided, rows, saturated)


def _second_divided_rows(eps, mu, tau, divided, rows, saturated):
    n = eps.numel()
    i = rows[:, None, None]
    k = torch.arange(n, device=eps.device)[None, :, None]
    j = torch.arange(n, device=eps.device)[None, None, :]
    return _second_divided_indices(eps, mu, tau, divided, i, k, j, saturated)


def _second_divided_indices(eps, mu, tau, divided, i, k, j, saturated):
    lo = torch.minimum(torch.minimum(i, k), j)
    hi = torch.maximum(torch.maximum(i, k), j)
    mid = i + k + j - lo - hi
    a, b, c = eps[lo], eps[mid], eps[hi]
    gap = c - a
    near = gap <= tau * 1e-3
    quotient = (divided[mid, hi] - divided[lo, mid]) / torch.where(
        near, torch.ones_like(gap), gap
    )
    del lo, hi, mid, gap
    if saturated:
        return torch.where(near, torch.zeros_like(quotient), quotient)
    center = (a + b + c) / 3
    spread = (a - center).square() + (b - center).square() + (c - center).square()
    del a, b, c
    f = torch.sigmoid((mu - center) / tau)
    del center
    p = f * (1 - f)
    # Complete homogeneous Taylor polynomial about the mean (linear term zero).
    local = p * (1 - 2 * f) / (2 * tau**2)
    local = local + p * (1 - 14 * f + 36 * f * f - 24 * f**3) * spread / (48 * tau**4)
    return torch.where(near, local, quotient)


def chunk_contraction(eps, mu, tau, divided, rows, a_rows, z_rows, a, z, saturated):
    """Evaluate and reduce one row tile; rows is a tensor to avoid offset guards."""
    t = _second_divided_rows(eps, mu, tau, divided, rows, saturated)
    return (t * (a_rows[:, :, None] * z[None, :, :]
                 + z_rows[:, :, None] * a[None, :, :])).sum(dim=1)


def chunk_contraction_last(eps, mu, tau, divided, rows, a_rows, z_rows, at, zt, saturated):
    """Construct [i,j,k] coefficients directly and reduce contiguous k."""
    indices = torch.arange(eps.numel(), device=eps.device)
    t = _second_divided_indices(
        eps, mu, tau, divided, rows[:, None, None],
        indices[None, None, :], indices[None, :, None], saturated,
    )
    return (t * (a_rows[:, None, :] * zt[None, :, :]
                 + z_rows[:, None, :] * at[None, :, :])).sum(dim=-1)


@lru_cache(maxsize=2)
def _compiled_chunk_contraction(reduction_last=False):
    # Lazy compilation; fail visibly instead of silently falling back to eager.
    return torch.compile(
        chunk_contraction_last if reduction_last else chunk_contraction,
        fullgraph=True, dynamic=True,
    )


def _setup(eps, f, mu, tau, count):
    saturated = (
        float(count).is_integer()
        and float((f * (1 - f)).sum()) <= 8 * torch.finfo(eps.dtype).eps
    )
    fp = -f * (1 - f) / tau
    if saturated:
        fp = torch.zeros_like(fp)
    total = fp.sum()
    weights = fp / torch.where(total != 0, total, torch.ones_like(total))
    return first_divided(eps, f, mu, tau, saturated), fp, weights, saturated


def response(U, divided, fp, weights, A):
    G = U.T @ ((A + A.T) / 2) @ U
    R = divided * G - torch.diag(fp * (weights * G.diagonal()).sum())
    out = U @ R @ U.T
    return (out + out.T) / 2


def second_response_contraction(
    U, eps, mu, tau, divided, fp, weights, A, Z, saturated=False, chunk_size=None, compile_chunk=False
):
    """Return C with C:E = A:D²P[E,Z], at fixed population."""
    a = U.T @ ((A + A.T) / 2) @ U
    z = U.T @ ((Z + Z.T) / 2) @ U
    eye = torch.eye(len(eps), device=eps.device, dtype=eps.dtype)
    a = a - eye * (weights * a.diagonal()).sum()
    z = z - eye * (weights * z.diagonal()).sum()
    size = len(eps) if chunk_size is None else chunk_size
    reduction_last = compile_chunk == "last"
    if reduction_last:
        a, z = a.contiguous(), z.contiguous()
        at, zt = a.T.contiguous(), z.T.contiguous()
    parts = []
    for start in range(0, len(eps), size):
        stop = min(start + size, len(eps))
        if compile_chunk:
            rows = torch.arange(start, stop, device=eps.device)
            parts.append(_compiled_chunk_contraction(reduction_last)(
                eps, mu, tau, divided, rows, a[start:stop], z[start:stop],
                at if reduction_last else a, zt if reduction_last else z, saturated
            ))
            continue
        t = second_divided(eps, mu, tau, divided, start, stop, saturated)
        parts.append(
            torch.einsum("ikj,ik,kj->ij", t, a[start:stop], z)
            + torch.einsum("ikj,ik,kj->ij", t, z[start:stop], a)
        )
    m = torch.cat(parts)
    m = m - torch.diag(weights * torch.trace(m))
    out = U @ m @ U.T
    return (out + out.T) / 2


class SpectralBackward(torch.autograd.Function):
    """Combined density/free-energy VJP with an explicit second derivative."""

    @staticmethod
    def forward(ctx, H, A, c, eps, U, f, mu, P, tau, count, chunk_size, compile_chunk=False):
        divided, fp, weights, saturated = _setup(eps, f, mu, tau, count)
        ctx.save_for_backward(A, c, eps, U, f, mu, P, tau, count)
        ctx.chunk_size = chunk_size
        ctx.compile_chunk = compile_chunk
        ctx.input_count = len(ctx.needs_input_grad)
        return response(U, divided, fp, weights, A) + c * P

    @staticmethod
    def backward(ctx, Z):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Spectral density supports second derivatives only (no third derivatives)"
            )
        A, c, eps, U, f, mu, P, tau, count = ctx.saved_tensors
        if float(count) in (0.0, float(eps.numel())):
            return (
                torch.zeros_like(A),
                torch.zeros_like(A),
                (Z * P).sum().reshape_as(c),
                *([None] * (ctx.input_count - 3)),
            )
        divided, fp, weights, saturated = _setup(eps, f, mu, tau, count)
        R = response(U, divided, fp, weights, Z)
        C = second_response_contraction(
            U, eps, mu, tau, divided, fp, weights, A, Z, saturated, ctx.chunk_size, ctx.compile_chunk
        )
        return C + c * R, R, (Z * P).sum().reshape_as(c), *([None] * (ctx.input_count - 3))
