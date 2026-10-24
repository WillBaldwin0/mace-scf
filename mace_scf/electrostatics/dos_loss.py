"""Spin-summed DOS supervision with occupied-spectrum alignment.

Only first parameter derivatives are supported. The alignment penalty centre is
intentionally detached; occupations retain their canonical population response.
"""
import hashlib
import math
from collections import OrderedDict

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit


def quadrature(x):
    dx = x[1:] - x[:-1]
    return torch.cat((dx[:1] / 2, (dx[:-1] + dx[1:]) / 2, dx[-1:] / 2))


def kernel(x, width, kind):
    """Density and CDF; width is Gaussian stddev or logistic scale, in eV."""
    z = x / width
    if kind == "gaussian":
        density = torch.exp(-0.5 * z.square()) / (width * math.sqrt(2 * math.pi))
        cumulative = 0.5 * torch.erfc(-z / math.sqrt(2))
    else:
        cumulative = torch.sigmoid(z)
        density = torch.sigmoid(z) * torch.sigmoid(-z) / width
    return density, cumulative


def numpy_kernel(x, width, kind):
    z = x / width
    if kind == "gaussian":
        return np.exp(-0.5 * z**2) / (width * math.sqrt(2 * math.pi))
    return expit(z) * expit(-z) / width


class DOSLoss(torch.nn.Module):
    def __init__(self, cumulative=False, energy_min=-5.0, energy_max=3.0,
                 grid_points=161, smearing="gaussian", width=0.2,
                 alignment_penalty=0.01, max_shift=5.0, search_points=41,
                 cache_size=1024):
        super().__init__()
        if smearing not in {"gaussian", "logistic"}:
            raise ValueError("DOS smearing must be gaussian or logistic")
        if not all(math.isfinite(x) for x in (energy_min, energy_max, width, alignment_penalty, max_shift)):
            raise ValueError("DOS loss settings must be finite")
        if not energy_min < 0 < energy_max or width <= 0 or alignment_penalty <= 0 or max_shift <= 0:
            raise ValueError("Require energy_min < 0 < energy_max and positive width, penalty and max_shift")
        if grid_points < 3 or search_points < 5 or cache_size < 0:
            raise ValueError("Invalid DOS grid, search or cache size")
        self.cumulative, self.smearing, self.width = cumulative, smearing, width
        self.energy_min, self.energy_max = energy_min, energy_max
        self.grid_points, self.search_points = grid_points, search_points
        self.alignment_penalty, self.max_shift = alignment_penalty, max_shift
        self.cache_size = cache_size
        self._reference_cache = OrderedDict()
        self.last_diagnostics = []

    def extra_repr(self):
        return (f"cumulative={self.cumulative}, smearing={self.smearing}, width={self.width}, "
                f"window=({self.energy_min}, {self.energy_max}), grid_points={self.grid_points}, "
                f"alignment_penalty={self.alignment_penalty}, max_shift={self.max_shift}")

    def reference(self, samples, mu, tau):
        """Cache stage-specific preparation; raw ragged samples remain in data."""
        raw = samples.detach().cpu().double().numpy()
        key = (hashlib.sha256(raw.tobytes()).digest(), float(mu), float(tau))
        if key not in self._reference_cache:
            x = torch.from_numpy(raw[:, 0] - float(mu))
            d = torch.from_numpy(raw[:, 1])
            grid = torch.linspace(self.energy_min, self.energy_max, self.grid_points, dtype=torch.float64)
            margin = (5 if self.smearing == "gaussian" else 12) * self.width
            if float(x[0]) > self.energy_min - margin or float(x[-1]) < self.energy_max + margin:
                raise ValueError("Reference DOS coverage is insufficient for the fit window and smoothing margin")
            # Occupy before broadening; tau is the DFT width, not model temperature.
            occupied = d * (torch.sigmoid(-x / float(tau)) if float(tau) > 0 else (x < 0).double())
            w = quadrature(x)
            k, cdf = kernel(grid[:, None] - x[None, :], self.width, self.smearing)
            full, occ = k @ (w * d), k @ (w * occupied)
            _, cdf_zero = kernel(-x, self.width, self.smearing)
            cumulative = (cdf - cdf_zero[None, :]) @ (w * d)
            result = (grid, quadrature(grid), full, occ, cumulative)
            if self.cache_size:
                self._reference_cache[key] = result
                if len(self._reference_cache) > self.cache_size:
                    self._reference_cache.popitem(last=False)
        else:
            result = self._reference_cache[key]
            self._reference_cache.move_to_end(key)
        return tuple(v.to(samples) for v in result)

    def align(self, eps, f, center, grid, weights, occupied):
        """Detached scalar solve. Refinement never records optimiser iterations."""
        e, pop, x, w, target = [v.detach().cpu().double().numpy() for v in (eps, f, grid, weights, occupied)]
        center = float(center)
        def objective(t):
            curve = numpy_kernel(x[:, None] + t - e[None, :], self.width, self.smearing) @ pop
            return float(np.dot(w, (curve - target)**2) + self.alignment_penalty * (t - center)**2)
        points = np.linspace(center - self.max_shift, center + self.max_shift, self.search_points)
        values = np.array([objective(t) for t in points])
        candidates = []
        for i in range(1, len(points)-1):
            if values[i] <= values[i-1] and values[i] <= values[i+1]:
                opt = minimize_scalar(objective, bounds=(points[i-1], points[i+1]),
                                      method="bounded", options={"xatol": 1e-11, "maxiter": 150})
                if opt.success:
                    candidates.append((opt.fun, opt.x))
        if not candidates:
            raise RuntimeError("DOS alignment has no interior minimum; increase max_shift or revise the window")
        candidates.sort()
        best, shift = candidates[0]
        if min(values[0], values[-1]) <= best:
            raise RuntimeError("DOS alignment minimum lies at the search boundary; increase max_shift")
        if len(candidates) > 1 and abs(candidates[1][0] - best) < 1e-9 * (1 + abs(best)):
            raise RuntimeError("DOS alignment has competing indistinguishable minima")
        return eps.new_tensor(shift), (points[0], points[-1])

    def one(self, channels, samples, mu_ref, tau_ref):
        grid, weights, full_ref, occ_ref, cumulative_ref = self.reference(samples, mu_ref, tau_ref)
        # RKS stores one channel representing both spins; UKS stores two.
        states = channels if len(channels) == 2 else channels * 2
        eps = torch.cat([c["energies"] for c in states])
        f = torch.cat([c["occupations"] for c in states])
        finite_mu = torch.stack([c["chemical_potential"] for c in states])
        finite_mu = finite_mu[torch.isfinite(finite_mu)]
        if not finite_mu.numel():
            raise ValueError("DOS alignment requires a finite model chemical potential")
        center = finite_mu.mean().detach()
        grid, weights, full_ref, occ_ref, cumulative_ref = [v.to(eps) for v in (grid, weights, full_ref, occ_ref, cumulative_ref)]
        t0, bounds = self.align(eps, f, center, grid, weights, occ_ref)
        # Refine stationarity in double precision on the model device. No graph
        # connects these Newton iterations to model parameters.
        with torch.enable_grad():
            def alignment(t, e, occ):
                k, _ = kernel(grid[:, None] + t - e[None, :], self.width, self.smearing)
                residual = k @ occ - occ_ref
                return (weights * residual.square()).sum() + self.alignment_penalty * (t-center).square()
            for _ in range(8):
                t = t0.detach().requires_grad_()
                a = alignment(t, eps.detach(), f.detach())
                slope = torch.autograd.grad(a, t, create_graph=True)[0]
                curvature = torch.autograd.grad(slope, t)[0]
                if not bool(torch.isfinite(curvature)) or float(curvature) <= 1e-10:
                    raise RuntimeError("DOS alignment has nonpositive or nearly zero curvature")
                if abs(float(slope)) <= max(1e-12, 100 * torch.finfo(eps.dtype).eps) * (1 + abs(float(a))):
                    break
                t0 = (t - slope / curvature).detach()
            else:
                raise RuntimeError("DOS alignment failed the stationarity check")
            if not bounds[0] < float(t) < bounds[1]:
                raise RuntimeError("DOS alignment refinement left the search interval")
            # Same forward optimum, explicit implicit first derivative. Autograd
            # computes mixed kernel derivatives, never an eigenvector derivative.
            t = t.detach().requires_grad_()
            a = alignment(t, eps, f)
            slope = torch.autograd.grad(a, t, create_graph=True)[0]
            aligned = t.detach() - (slope - slope.detach()) / curvature.detach()
            k, cdf = kernel(grid[:, None] + aligned - eps[None, :], self.width, self.smearing)
            if self.cumulative:
                _, cdf_zero = kernel(aligned - eps, self.width, self.smearing)
                prediction = (cdf - cdf_zero[None, :]).sum(-1)
                target = cumulative_ref
            else:
                prediction, target = k.sum(-1), full_ref
            loss = (weights * (prediction-target).square()).sum() / (self.energy_max-self.energy_min)
        self.last_diagnostics.append(dict(shift=float(t.detach()),
            relative_shift=float(t.detach()-center), alignment=float(a.detach()),
            curvature=float(curvature.detach()), mse=float(loss.detach())))
        return loss

    def forward(self, ref, pred):
        self.last_diagnostics = []
        sizes = ref.dos_num_points.reshape(-1).tolist()
        values = []
        offset = 0
        for g, size in enumerate(sizes):
            weight = ref.weight.reshape(-1)[g] * ref.dos_weight.reshape(-1)[g]
            if float(weight) > 0:
                if "dos_state" not in pred:
                    raise ValueError("DOS loss requires a one-shot model with spectral loss outputs")
                value = self.one(pred["dos_state"][g], ref.dos_samples[offset:offset+size],
                                 ref.dos_chemical_potential.reshape(-1)[g], ref.dos_elec_temp.reshape(-1)[g])
                values.append(weight.to(value) * value)
            offset += size
        if not values:
            return pred["energy"].sum() * 0
        return torch.stack(values).sum() / len(sizes)
