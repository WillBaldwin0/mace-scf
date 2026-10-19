"""Tensor-only chemical-potential bisection, without derivative corrections."""

from functools import lru_cache

import torch


def bisect_chemical_potential(eps, count, tau):
    lo, hi = eps.min() - 80 * tau, eps.max() + 80 * tau
    for _ in range(80):
        mu = (lo + hi) / 2
        below = torch.sigmoid((mu - eps) / tau).sum() < count
        lo = torch.where(below, mu, lo)
        hi = torch.where(below, hi, mu)
    return (lo + hi) / 2


@lru_cache(maxsize=1)
def compiled_bisection():
    return torch.compile(bisect_chemical_potential, fullgraph=True, dynamic=True)
