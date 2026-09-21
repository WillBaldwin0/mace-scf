"""Equivariant s/p Hamiltonians and fixed-population electronic states."""

import math
from typing import Optional

import torch
from e3nn import o3
from torch import nn


class SPBlockReadout(nn.Module):
    """Learn coefficients, then decode each item into an unrestricted B x B block.

    Items may be directed edges or atoms. Ns counts s orbitals; Np counts p
    triplets. Orbital order is [all s orbitals, p0_xyz, p1_xyz, ...].
    """

    def __init__(self, feature_irreps, n_s: int, n_p: int):
        super().__init__()
        if n_s < 0 or n_p < 0 or n_s + n_p == 0:
            raise ValueError("Require nonnegative shell counts and a nonempty basis")
        self.n_s, self.n_p = n_s, n_p
        self.block_size = n_s + 3 * n_p
        self.feature_irreps = o3.Irreps(feature_irreps)

        # Explicit order: S, T, V, W, Z, Q. Do not sort this representation:
        # forward() splits its flat output in exactly this order.
        sectors = [
            (n_s * n_s, (0, 1)),  # S: s-s scalars
            (n_p * n_p, (0, 1)),  # T: p-p trace scalars
            (n_s * n_p, (1, -1)),  # V: s-p polar vectors
            (n_p * n_s, (1, -1)),  # W: p-s polar vectors
            (n_p * n_p, (1, 1)),  # Z: p-p axial vectors
            (n_p * n_p, (2, 1)),  # Q: p-p symmetric-traceless tensors
        ]
        self.coefficient_irreps = o3.Irreps(
            [(mul, ir) for mul, ir in sectors if mul > 0]
        )
        self.split_sizes = [mul * (2 * ir[0] + 1) for mul, ir in sectors]
        assert self.coefficient_irreps.dim == self.block_size**2

        # A Linear cannot manufacture a missing angular/parity sector.
        required = {ir for _, ir in self.coefficient_irreps}
        available = {ir for mul, ir in self.feature_irreps if mul > 0}
        if not required.issubset(available):
            raise ValueError("Input features are missing a required coefficient irrep")

        # One learned map, with independent weights for the output copies.
        # Leading item axes are NOT matrix-channel axes.
        self.readout = o3.Linear(
            self.feature_irreps, self.coefficient_irreps, biases=False
        )

        # Fixed, small angular bases. The module's .to(device/dtype) moves these.
        self.register_buffer("identity", torch.eye(3))
        self.register_buffer("quadrupole_basis", 5**0.5 * o3.wigner_3j(2, 1, 1))
        epsilon = torch.zeros(3, 3, 3)
        epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1.0
        epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1.0
        # Axes: [axial component, row Cartesian, column Cartesian].
        self.register_buffer("axial_basis", epsilon.permute(2, 0, 1) / 2**0.5)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [items, feature_dim]; return: [items, B, B]."""
        if features.ndim != 2 or features.shape[-1] != self.feature_irreps.dim:
            raise ValueError("Expected [items, feature_irreps.dim] features")
        count = features.shape[0]
        ns, np = self.n_s, self.n_p
        x = self.readout(features)  # [items, B**2], flattened irrep layout
        S, T, V, W, Z, Q = torch.split(x, self.split_sizes, dim=-1)

        S = S.reshape(count, ns, ns)
        T = T.reshape(count, np, np)
        V = V.reshape(count, ns, np, 3)
        W = W.reshape(count, np, ns, 3)
        Z = Z.reshape(count, np, np, 3)
        Q = Q.reshape(count, np, np, 5)

        # p-p axes are [item, row shell, row xyz, column shell, column xyz].
        pp = torch.einsum("euv,ab->euavb", T, self.identity) / 3**0.5
        pp = pp + torch.einsum("euvk,kab->euavb", Z, self.axial_basis)
        pp = pp + torch.einsum("euvk,kab->euavb", Q, self.quadrupole_basis)
        pp = pp.reshape(count, 3 * np, 3 * np)

        sp = V.reshape(count, ns, 3 * np)
        ps = W.permute(0, 1, 3, 2).reshape(count, 3 * np, ns)
        top = torch.cat((S, sp), dim=-1)
        bottom = torch.cat((ps, pp), dim=-1)
        return torch.cat((top, bottom), dim=-2)


def assemble_matrix(
    edge_blocks: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    onsite_blocks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scatter a full directed edge list, symmetrize, and insert on-site blocks.

    edge_blocks: [E, B, B]; edge_index: [2, E] int64, on the same device.
    Every edge must have its reverse. Periodic self-image edges are allowed.
    Reverse-edge completeness is an input contract, not checked here.
    Repeated entries sum, so unintended duplicate edges must be removed upstream.
    onsite_blocks: optional [N, B, B]; absent means zero on-site blocks.
    Return [N*B, N*B]. All floating tensors must share dtype and device.
    """
    if num_nodes <= 0:
        raise ValueError("Expected a nonempty structure")
    if edge_index.shape != (2, edge_blocks.shape[0]):
        raise ValueError("edge_index and edge_blocks disagree on edge count")
    sender, receiver = edge_index
    B = edge_blocks.shape[-1]

    # Scatter into ordered atom pairs; index_add keeps autograd through blocks.
    flat_blocks = edge_blocks.new_zeros(num_nodes * num_nodes, B, B)
    flat_blocks = flat_blocks.index_add(0, sender * num_nodes + receiver, edge_blocks)
    blocks = flat_blocks.reshape(num_nodes, num_nodes, B, B)
    M = blocks.permute(0, 2, 1, 3).reshape(num_nodes * B, num_nodes * B)

    # One forward pass over the full directed list already evaluated +/- r.
    H = 0.5 * (M + M.transpose(-1, -2))
    if onsite_blocks is not None:
        if onsite_blocks.shape != (num_nodes, B, B):
            raise ValueError("Expected on-site blocks with shape [N, B, B]")
        D = 0.5 * (onsite_blocks + onsite_blocks.transpose(-1, -2))
        # Add to any periodic self-image contributions on the diagonal.
        for i in range(num_nodes):
            H[i * B : (i + 1) * B, i * B : (i + 1) * B] = (
                H[i * B : (i + 1) * B, i * B : (i + 1) * B] + D[i]
            )
    return H


class MatrixConstruction(nn.Module):
    """Glue for the two readouts and assembly; the upstream model is external."""

    def __init__(self, edge_irreps, n_s: int, n_p: int, onsite_irreps=None):
        super().__init__()
        self.edge_readout = SPBlockReadout(edge_irreps, n_s, n_p)
        # Deliberately simple: unrestricted on-site coefficients followed by
        # transpose projection. A minimal symmetric head could remove redundancy.
        self.onsite_readout = (
            SPBlockReadout(onsite_irreps, n_s, n_p)
            if onsite_irreps is not None
            else None
        )

    def forward(self, edge_features, edge_index, num_nodes, onsite_features=None):
        """Readouts have independent parameters; all edges share the edge head."""
        edge_blocks = self.edge_readout(edge_features)
        onsite_blocks = None
        if self.onsite_readout is not None:
            if onsite_features is None:
                raise ValueError(
                    "This module was configured to require on-site features"
                )
            onsite_blocks = self.onsite_readout(onsite_features)
        elif onsite_features is not None:
            raise ValueError("Configure onsite_irreps to use on-site features")
        return assemble_matrix(edge_blocks, edge_index, num_nodes, onsite_blocks)


class HamiltonianBuilder(nn.Module):
    """Two equivariant heads; a real Gamma-point Hamiltonian for each graph.

    Node features use MACE's (y,z,x) frame; output orbitals use (px,py,pz).
    A full directed neighbour list, including reverse image shifts, is required.
    """

    def __init__(
        self,
        node_irreps,
        n_s=2,
        n_p=3,
        edge_mode="bilinear",
        r_max=3.0,
        num_radial=8,
        feature_multiplicity=4,
        radial_hidden=32,
        onsite_mode="quadratic",
    ):
        super().__init__()
        if edge_mode not in {"bilinear", "linear_endpoints"}:
            raise ValueError("Unknown edge mode")
        self.edge_mode = edge_mode
        self.node_irreps = o3.Irreps(node_irreps)
        self.feature_irreps = o3.Irreps(
            [(feature_multiplicity, ir) for ir in ("0e", "1o", "1e", "2e")]
        )
        if r_max <= 0 or num_radial < 1 or feature_multiplicity < 1:
            raise ValueError("Require positive cutoff and feature sizes")
        if not {o3.Irrep("0e"), o3.Irrep("1o")}.issubset(
            {ir for mul, ir in self.node_irreps if mul}
        ):
            raise ValueError(
                "Hamiltonian heads require scalar and polar-vector node features"
            )
        if onsite_mode not in {"quadratic", "direct"}:
            raise ValueError("onsite_mode must be quadratic or direct")
        self.onsite_mode = onsite_mode
        self.sh_irreps = o3.Irreps.spherical_harmonics(1)
        self.r_max = r_max
        self.register_buffer("radial_centres", torch.linspace(0, r_max, num_radial))

        if edge_mode == "bilinear":
            self.pair_tp = o3.FullyConnectedTensorProduct(
                self.node_irreps, self.node_irreps, self.node_irreps
            )
            self.pair_linear = o3.Linear(self.node_irreps, self.node_irreps)
        else:
            self.sender_linear = o3.Linear(self.node_irreps, self.node_irreps)
            self.receiver_linear = o3.Linear(self.node_irreps, self.node_irreps)

        self.angular_tp = o3.FullyConnectedTensorProduct(
            self.node_irreps,
            self.sh_irreps,
            self.feature_irreps,
            internal_weights=False,
            shared_weights=False,
        )
        self.radial_mlp = nn.Sequential(
            nn.Linear(num_radial, radial_hidden),
            nn.SiLU(),
            nn.Linear(radial_hidden, self.angular_tp.weight_numel),
        )

        # Independent on-site factors permit cross-channel axial features.
        self.onsite_left = o3.Linear(self.node_irreps, self.node_irreps)
        self.onsite_right = o3.Linear(self.node_irreps, self.node_irreps)
        self.onsite_tp = o3.FullyConnectedTensorProduct(
            self.node_irreps, self.node_irreps, self.feature_irreps
        )
        self.matrix = MatrixConstruction(
            self.feature_irreps,
            n_s,
            n_p,
            self.feature_irreps if onsite_mode == "quadratic" else self.node_irreps,
        )
        self.block_size = n_s + 3 * n_p
        # Raw decoded p components follow feature-frame coordinates (y,z,x).
        # The density map expects Cartesian orbital order (px,py,pz).
        order = list(range(n_s))
        for shell in range(n_p):
            order.extend(n_s + 3 * shell + j for j in (2, 0, 1))
        self.register_buffer("orbital_order", torch.tensor(order, dtype=torch.long))

    def forward(self, node_features, positions, edge_index, shifts=None, ptr=None):
        sender, receiver = edge_index
        vectors = positions[receiver] - positions[sender]
        if shifts is not None:
            vectors = vectors + shifts
        if torch.any(vectors.norm(dim=-1) == 0):
            raise ValueError("Zero-length Hamiltonian edges are not allowed")
        distances = vectors.norm(dim=-1)
        sh = o3.spherical_harmonics(
            self.sh_irreps,
            vectors[:, [1, 2, 0]],
            normalize=True,
            normalization="component",
        )
        radial = torch.exp(
            -(((distances[:, None] - self.radial_centres) / self.r_max) ** 2)
        )
        x = (distances / self.r_max).clamp(0, 1)
        cutoff = 1 - 10 * x**3 + 15 * x**4 - 6 * x**5
        if self.edge_mode == "bilinear":
            pair = self.pair_linear(
                self.pair_tp(node_features[sender], node_features[receiver])
            )
        else:
            left = self.sender_linear(node_features)
            right = self.receiver_linear(node_features)
            pair = left[sender] + right[receiver]
        edge_features = self.angular_tp(pair, sh, self.radial_mlp(radial))
        edge_features = edge_features * cutoff[:, None]
        onsite_features = (
            self.onsite_tp(
                self.onsite_left(node_features), self.onsite_right(node_features)
            )
            if self.onsite_mode == "quadratic"
            else node_features
        )
        n = positions.shape[0]
        if ptr is None:
            ptr = torch.tensor([0, n], device=positions.device)
        blocks = self.matrix.edge_readout(edge_features)
        onsite = self.matrix.onsite_readout(onsite_features)
        matrices = []
        used = torch.zeros(sender.shape, dtype=torch.bool, device=sender.device)
        for first, last in zip(ptr[:-1].tolist(), ptr[1:].tolist()):
            mask = (
                (sender >= first)
                & (sender < last)
                & (receiver >= first)
                & (receiver < last)
            )
            used |= mask
            raw = assemble_matrix(
                blocks[mask],
                edge_index[:, mask] - first,
                last - first,
                onsite[first:last],
            )
            order = (
                torch.arange(last - first, device=positions.device)[:, None]
                * self.block_size
                + self.orbital_order
            ).flatten()
            matrices.append(raw.index_select(0, order).index_select(1, order))
        if not bool(used.all()):
            raise ValueError("Hamiltonian edges must not connect different graphs")
        return matrices


def _fill(eps, count, tau, degeneracy_tolerance=1e-9):
    """Canonical occupations and chemical potential; differentiable root correction."""
    n = eps.numel()
    target = float(count.detach())
    if not 0 <= target <= n:
        raise ValueError(f"Spin population {target} is outside [0, {n}]")
    if not bool(torch.isfinite(tau)) or float(tau.detach()) < 0:
        raise ValueError("Electronic temperature must be finite and nonnegative")
    if target == 0 or target == n:
        f = eps * 0 + (target == n)
        mu = eps.new_tensor(float("-inf") if target == 0 else float("inf"))
        return f, mu
    if float(tau.detach()) == 0:
        # Equal filling within each degenerate level, including fractional counts.
        with torch.no_grad():
            f = torch.zeros_like(eps)
            remaining = target
            start = 0
            while start < n:
                end = start + 1
                while (
                    end < n
                    and abs(float(eps[end] - eps[start])) <= degeneracy_tolerance
                ):
                    end += 1
                occupation = min(1.0, max(0.0, remaining / (end - start)))
                f[start:end] = occupation
                remaining -= occupation * (end - start)
                start = end
            partial = (f > 0) & (f < 1)
            if bool(partial.any()):
                mu = eps[partial].mean()
            else:
                last = int(target) - 1
                mu = (eps[last] + eps[last + 1]) / 2
        return f + eps * 0, mu
    # An integer insulating count has a broad numerical root plateau. Keep
    # mu in the middle of the gap, away from either occupation edge.
    if target.is_integer() and 0 < target < n:
        middle = (eps[int(target) - 1] + eps[int(target)]) / 2
        trial = torch.sigmoid((middle - eps) / tau)
        if (
            float((trial * (1 - trial)).sum().detach())
            <= 8 * torch.finfo(eps.dtype).eps
        ):
            return trial, middle
    # Bisection is only a root finder. Newton corrections restore implicit
    # derivatives of the fixed-count root, including those needed by force loss.
    with torch.no_grad():
        lo, hi = eps.min() - 80 * tau, eps.max() + 80 * tau
        for _ in range(80):
            mu = (lo + hi) / 2
            below = torch.sigmoid((mu - eps) / tau).sum() < count
            lo = torch.where(below, mu, lo)
            hi = torch.where(below, hi, mu)
        mu = (lo + hi) / 2
    for _ in range(3):
        f = torch.sigmoid((mu - eps) / tau)
        slope = (f * (1 - f)).sum() / tau
        safe = slope.clamp_min(torch.finfo(eps.dtype).tiny)
        mu = mu + torch.where(slope > 0, (count - f.sum()) / safe, torch.zeros_like(mu))
    return torch.sigmoid((mu - eps) / tau), mu


class FermiDiracOccupations(nn.Module):
    """Per-spin fixed electron counts. Energies are eV; temperature is K or eV."""

    def __init__(self, elec_temp_units="kelvin", degeneracy_tolerance=1e-9):
        super().__init__()
        if elec_temp_units not in {"kelvin", "eV"}:
            raise ValueError("elec_temp_units must be kelvin or eV")
        self.elec_temp_units = elec_temp_units
        self.degeneracy_tolerance = degeneracy_tolerance

    def thermal_energy(self, temperature):
        return temperature * (
            8.617333262145e-5 if self.elec_temp_units == "kelvin" else 1.0
        )

    def forward(self, eigenvalues, populations, elec_temp):
        tau = self.thermal_energy(elec_temp)
        fs, mus = zip(
            *[
                _fill(eigenvalues, count, tau, self.degeneracy_tolerance)
                for count in populations
            ]
        )
        f = torch.stack(fs, dim=-1)
        mu = torch.stack(mus)
        entropy = eigenvalues.sum() * 0
        if float(tau.detach()) > 0:
            for s in range(2):
                if bool(torch.isfinite(mu[s])):
                    x = (mu[s] - eigenvalues) / tau
                    entropy = (
                        entropy
                        + tau
                        * (
                            fs[s] * torch.nn.functional.logsigmoid(x)
                            + (1 - fs[s]) * torch.nn.functional.logsigmoid(-x)
                        ).sum()
                    )
        return f, mu, entropy


def _smooth_density(H, count, tau, eps, mu, occupations):
    """Matrix Fermi density and chemical potential from a saved spectral state.

    Only used while recording derivatives. Saturated insulating channels keep
    their gap-midpoint chemical potential but retain subspace rotation.
    """
    eye = torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
    target = float(count.detach())
    if target in (0, H.shape[0]):
        return H * 0 + eye * (target != 0) + (count + tau) * 0, mu
    precision = torch.finfo(H.dtype).eps
    count_tol = 128 * precision * H.shape[0]
    saturated = (
        target.is_integer()
        and float((occupations * (1 - occupations)).sum()) <= 8 * precision
    )
    with torch.no_grad():
        bound = float(((eps - mu) / tau).abs().max())
        steps = max(0, int(math.ceil(math.log2(max(bound, 1.0)))))

    def evaluate(chemical_potential):
        exponent = (H - chemical_potential * eye) / (tau * 2**steps)
        F = torch.linalg.solve(eye + torch.matrix_exp(exponent), eye)
        for _ in range(steps):
            square = F @ F
            complement = eye - F
            F = torch.linalg.solve(square + complement @ complement, square)
            F = (F + F.T) / 2
        return (F + F.T) / 2

    for _ in range(3):
        F = evaluate(mu)
        residual = count - torch.trace(F)
        susceptibility = torch.trace(F - F @ F)
        if saturated or float(susceptibility.detach()) <= 8 * precision:
            if abs(float(residual.detach())) > count_tol:
                raise RuntimeError(
                    "Insulating density reconstruction has incorrect electron count"
                )
            return F + count * 0, mu
        mu = mu + residual * tau / susceptibility
    F = evaluate(mu)
    if abs(float((count - torch.trace(F)).detach())) > count_tol:
        raise RuntimeError(
            "Smooth density chemical-potential corrections did not converge"
        )
    return F, mu


def _spectral_response(U, eps, f, tau, grad, tolerance):
    """Fixed-count density VJP from cached eigenpairs; no eigensolve."""
    G = U.T @ ((grad + grad.T) / 2) @ U
    gaps = eps[:, None] - eps[None, :]
    close = gaps.abs() <= tolerance
    safe_gaps = torch.where(close, torch.ones_like(gaps), gaps)
    fp = -f * (1 - f) / tau if float(tau) > 0 else torch.zeros_like(f)
    divided = torch.where(
        close, (fp[:, None] + fp[None, :]) / 2, (f[:, None] - f[None, :]) / safe_gaps
    )
    total = fp.sum()
    safe_total = torch.where(total != 0, total, torch.ones_like(total))
    weighted = (fp * G.diagonal()).sum() / safe_total
    response = U @ (divided * G - torch.diag(fp * weighted)) @ U.T
    width = tau * 0
    if float(tau) > 0:
        mean_eps = (fp * eps).sum() / safe_total
        width = (G.diagonal() * fp * (mean_eps - eps) / tau).sum()
    return (response + response.T) / 2, weighted, width


class _CanonicalState(torch.autograd.Function):
    """Shared-spin spectral state, with cached first and smooth second derivatives."""

    @staticmethod
    def forward(ctx, H, counts, tau, tolerance):
        eps, U = torch.linalg.eigh(H)
        fs, mus, entropies, densities = [], [], [], []
        for spin in range(2):
            if spin and bool(counts[0] == counts[1]):
                f, mu = fs[0], mus[0]
            else:
                f, mu = _fill(eps, counts[spin], tau, tolerance)
            entropy = tau * 0
            if float(tau) > 0 and bool(torch.isfinite(mu)):
                x = (mu - eps) / tau
                entropy = (
                    f * torch.nn.functional.logsigmoid(x)
                    + (1 - f) * torch.nn.functional.logsigmoid(-x)
                ).sum()
            fs.append(f)
            mus.append(mu)
            entropies.append(entropy)
            densities.append(
                densities[0] if spin and bool(counts[0] == counts[1]) else (U * f) @ U.T
            )
        f, mu, entropy = torch.stack(fs, -1), torch.stack(mus), torch.stack(entropies)
        P = torch.stack(densities)
        free = (f * eps[:, None]).sum() + tau * entropy.sum()
        ctx.save_for_backward(H, counts, tau, eps, U, f, mu, entropy, P)
        ctx.tolerance = tolerance
        ctx.mark_non_differentiable(eps, f, mu)
        ctx.set_materialize_grads(False)
        return P, free, eps, f, mu

    @staticmethod
    def backward(ctx, grad_P, grad_free, *unused):
        H, counts, tau, eps, U, f, mu, entropy, saved_P = ctx.saved_tensors
        if grad_P is None:
            grad_P = H.new_zeros((2,) + H.shape)
        if grad_free is None:
            grad_free = H.new_zeros(())
        if torch.is_grad_enabled():
            if float(tau.detach()) == 0:
                raise RuntimeError(
                    "Force gradients require positive electronic smearing"
                )
            inputs = tuple(
                t if t.requires_grad else t.detach().requires_grad_(True)
                for t in (H, counts, tau)
            )
            h, populations, width = inputs
            shared = not counts.requires_grad and bool(counts[0] == counts[1])
            densities, potentials = [], []
            for spin in range(2):
                if spin and shared:
                    density, potential = densities[0], potentials[0]
                else:
                    density, potential = _smooth_density(
                        h, populations[spin], width, eps, mu[spin], f[:, spin]
                    )
                densities.append(density)
                potentials.append(potential)
            P = torch.stack(densities)
            dH, dN, dt = torch.autograd.grad(P, inputs, grad_P, create_graph=True)
            # Entropy value plus its exact local differential, sufficient for
            # double backward without a matrix logarithm or another eigensolve.
            ds = []
            for spin in range(2):
                if not bool(torch.isfinite(mu[spin])):
                    ds.append(width * 0)
                else:
                    eye = torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
                    coefficient = ((h - potentials[spin] * eye) / width).detach()
                    delta = densities[spin] - densities[spin].detach()
                    ds.append(entropy[spin] - (coefficient * delta).sum())
            finite_mu = torch.stack(
                [
                    potential if bool(torch.isfinite(mu[i])) else populations[i] * 0
                    for i, potential in enumerate(potentials)
                ]
            )
            return (
                dH + grad_free * P.sum(0),
                dN + grad_free * finite_mu,
                dt + grad_free * torch.stack(ds).sum(),
                None,
            )
        if not counts.requires_grad and bool(counts[0] == counts[1]):
            response, _, width = _spectral_response(
                U, eps, f[:, 0], tau, grad_P.sum(0), ctx.tolerance
            )
            return (
                response + grad_free * saved_P.sum(0),
                None,
                width + grad_free * entropy.sum(),
                None,
            )
        dH, dN, dt = torch.zeros_like(H), [], tau * 0
        for spin in range(2):
            response, population, width = _spectral_response(
                U, eps, f[:, spin], tau, grad_P[spin], ctx.tolerance
            )
            dH = dH + response + grad_free * saved_P[spin]
            potential = mu[spin] if bool(torch.isfinite(mu[spin])) else mu.new_zeros(())
            dN.append(population + grad_free * potential)
            dt = dt + width + grad_free * entropy[spin]
        return dH, torch.stack(dN), dt, None


class ElectronicState(nn.Module):
    """Solve independent dense Hamiltonians and return on-site density blocks."""

    def __init__(self, block_size, elec_temp_units="kelvin", degeneracy_tolerance=1e-9):
        super().__init__()
        self.block_size = block_size
        self.occupations = FermiDiracOccupations(elec_temp_units, degeneracy_tolerance)

    def forward(self, hamiltonians, N_alpha, N_beta, elec_temp):
        gamma, spin, bands, entropies, mus, residuals, values, fillings = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for g, H in enumerate(hamiltonians):
            counts = torch.stack((N_alpha[g], N_beta[g]))
            tau = self.occupations.thermal_energy(elec_temp[g])
            densities, free_band, eps, f, mu = _CanonicalState.apply(
                H, counts, tau, self.occupations.degeneracy_tolerance
            )
            total, difference = densities[0] + densities[1], densities[0] - densities[1]
            B = self.block_size
            gamma.extend(total[i : i + B, i : i + B] for i in range(0, H.shape[0], B))
            spin.extend(
                difference[i : i + B, i : i + B] for i in range(0, H.shape[0], B)
            )
            bands.append((total * H).sum())
            entropies.append(free_band - bands[-1])
            mus.append(mu)
            residuals.append(f.sum(0) - counts.detach())
            values.append(eps)
            fillings.append(f)
        return dict(
            gamma=torch.stack(gamma),
            gamma_spin=torch.stack(spin),
            band_energy=torch.stack(bands),
            entropy_energy=torch.stack(entropies),
            chemical_potentials=torch.stack(mus),
            count_residuals=torch.stack(residuals),
            eigenvalues=values,
            occupations=fillings,
        )
