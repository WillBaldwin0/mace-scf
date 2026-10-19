"""Equivariant s/p Hamiltonians and fixed-population electronic states."""

import math

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
        return self.decode(self.readout(features))

    def decode(self, x):
        """Decode coefficients in the fixed S,T,V,W,Z,Q order."""
        count = x.shape[0]
        ns, np = self.n_s, self.n_p
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


def _xc_readout(size, readout, hidden_features):
    if readout == "linear":
        return nn.Linear(size, 1)
    if readout == "nonlinear":
        if not isinstance(hidden_features, int) or hidden_features < 1:
            raise ValueError("XC hidden_features must be a positive integer")
        return nn.Sequential(
            nn.Linear(size, hidden_features), nn.SiLU(), nn.Linear(hidden_features, 1)
        )
    raise ValueError("XC readout must be linear or nonlinear")


class _DensityBlockIrreps(nn.Module):
    """Fixed Cartesian s/p matrix decomposition into MACE-frame irreps."""

    def __init__(self, n_s, n_p):
        super().__init__()
        if min(n_s, n_p) < 0 or n_s + n_p == 0:
            raise ValueError("Require a nonempty s/p basis")
        self.n_s, self.n_p = n_s, n_p
        self.block_size = n_s + 3 * n_p
        self.irreps = o3.Irreps(
            [
                (m, ir)
                for m, ir in (
                    (n_s * n_s + n_p * n_p, "0e"),
                    (2 * n_s * n_p, "1o"),
                    (n_p * n_p, "1e"),
                    (n_p * n_p, "2e"),
                )
                if m
            ]
        )
        self.register_buffer("to_feature_order", _orbital_order(n_s, n_p).argsort())
        self.register_buffer("quadrupole_basis", 5**0.5 * o3.wigner_3j(2, 1, 1))
        epsilon = torch.zeros(3, 3, 3)
        epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
        epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1
        self.register_buffer("axial_basis", epsilon / math.sqrt(2))

    def forward(self, density):
        if density.ndim != 3 or density.shape[1:] != (self.block_size, self.block_size):
            raise ValueError("Expected on-site density blocks [N,B,B]")
        density = (density + density.transpose(-1, -2)) / 2
        density = density.index_select(1, self.to_feature_order).index_select(
            2, self.to_feature_order
        )
        n, s, p = density.shape[0], self.n_s, self.n_p
        pp = density[:, s:, s:].reshape(n, p, 3, p, 3)
        scalar = torch.cat(
            (
                density[:, :s, :s].reshape(n, s * s),
                torch.einsum("nuava->nuv", pp).reshape(n, p * p) / math.sqrt(3),
            ),
            -1,
        )
        polar = torch.cat(
            (
                density[:, :s, s:].reshape(n, s * p * 3),
                density[:, s:, :s]
                .reshape(n, p, 3, s)
                .permute(0, 1, 3, 2)
                .reshape(n, p * s * 3),
            ),
            -1,
        )
        axial = torch.einsum("nuavb,abk->nuvk", pp, self.axial_basis).reshape(
            n, p * p * 3
        )
        quadrupole = torch.einsum("nuavb,kab->nuvk", pp, self.quadrupole_basis).reshape(
            n, p * p * 5
        )
        return torch.cat((scalar, polar, axial, quadrupole), -1)


class OnsiteExchangeCorrelation(nn.Module):
    """Spin-even atomic energy from full on-site matrices and local features.

    Inputs are alpha+beta and alpha-beta matrices in Cartesian orbital order.
    All density derivatives remain available; no effective potential is computed.
    """

    def __init__(
        self,
        n_s,
        n_p,
        node_irreps,
        readout="nonlinear",
        invariant_channels=16,
        hidden_features=64,
    ):
        super().__init__()
        if not isinstance(invariant_channels, int) or invariant_channels < 1:
            raise ValueError("XC invariant_channels must be a positive integer")
        self.decomposition = _DensityBlockIrreps(n_s, n_p)
        self.node_irreps = o3.Irreps(node_irreps)
        self.density_irreps = o3.Irreps(
            [(invariant_channels, ir) for _, ir in self.decomposition.irreps]
        )
        self.total_mix = o3.Linear(
            self.decomposition.irreps, self.density_irreps, biases=False
        )
        self.spin_mix = o3.Linear(
            self.decomposition.irreps, self.density_irreps, biases=False
        )
        available = {ir for mul, ir in self.node_irreps if mul}
        self.matching = [ir in available for _, ir in self.density_irreps]
        matching_irreps = o3.Irreps(
            [
                (mul, ir)
                for (mul, ir), match in zip(self.density_irreps, self.matching)
                if match
            ]
        )
        self.node_projection = o3.Linear(
            self.node_irreps, matching_irreps, biases=False
        )
        self.channels = invariant_channels
        self.num_scalar = self.decomposition.irreps.count(o3.Irrep("0e"))
        odd_size = sum(self.matching) * invariant_channels
        self.spin_invariant_mix = (
            nn.Linear(odd_size, invariant_channels, bias=False) if odd_size else None
        )
        # Raw total scalars, mixed density norms, total-node scalars, squared spin-node scalars.
        size = (
            self.num_scalar
            + 2 * len(self.density_irreps) * invariant_channels
            + odd_size
        )
        if odd_size:
            size += invariant_channels
        self.readout = _xc_readout(size, readout, hidden_features)

    def invariants(self, gamma, gamma_spin, node_features):
        if gamma_spin.shape != gamma.shape or node_features.shape != (
            gamma.shape[0],
            self.node_irreps.dim,
        ):
            raise ValueError("XC density/node shapes disagree")
        total = self.decomposition(gamma)
        spin = self.decomposition(gamma_spin)
        q, m = self.total_mix(total), self.spin_mix(spin)
        nodes = self.node_projection(node_features)
        features, total_node, spin_node = [total[:, : self.num_scalar]], [], []
        start, node_start = 0, 0
        for (mul, ir), match in zip(self.density_irreps, self.matching):
            end = start + mul * ir.dim
            qi = q[:, start:end].reshape(-1, mul, ir.dim)
            mi = m[:, start:end].reshape(-1, mul, ir.dim)
            features.extend((qi.square().sum(-1), mi.square().sum(-1)))
            if match:
                hi = nodes[:, node_start : node_start + mul * ir.dim].reshape(
                    -1, mul, ir.dim
                )
                total_node.append((qi * hi).sum(-1))
                spin_node.append((mi * hi).sum(-1))
                node_start += mul * ir.dim
            start = end
        if spin_node:
            features.extend(total_node)
            features.append(self.spin_invariant_mix(torch.cat(spin_node, -1)).square())
        return torch.cat(features, -1)

    def forward(self, gamma, gamma_spin, node_features):
        return self.readout(self.invariants(gamma, gamma_spin, node_features)).squeeze(
            -1
        )


class MultipoleExchangeCorrelation(nn.Module):
    """Restricted spin-even energy from electronic monopoles/dipoles only.

    Inputs [N,4] are electron-alpha+beta and electron-alpha-beta, ordered
    (monopole, dipole_y, dipole_z, dipole_x), without nuclear charge.
    """

    def __init__(
        self,
        node_irreps,
        readout="nonlinear",
        invariant_channels=16,
        hidden_features=64,
        norm_epsilon=0.01,
    ):
        super().__init__()
        if not isinstance(invariant_channels, int) or invariant_channels < 1:
            raise ValueError("XC invariant_channels must be a positive integer")
        if not math.isfinite(norm_epsilon) or norm_epsilon <= 0:
            raise ValueError("XC norm_epsilon must be positive and finite")
        self.node_irreps = o3.Irreps(node_irreps)
        self.norm_epsilon = float(norm_epsilon)
        self.matches = [
            (sl, mul, ir)
            for sl, (mul, ir) in zip(self.node_irreps.slices(), self.node_irreps)
            if ir in (o3.Irrep("0e"), o3.Irrep("1o")) and mul
        ]
        count = sum(mul for _, mul, _ in self.matches)
        self.spin_invariant_mix = (
            nn.Linear(count, invariant_channels, bias=False) if count else None
        )
        self.readout = _xc_readout(
            4 + count + (invariant_channels if count else 0), readout, hidden_features
        )

    def invariants(self, electron_multipoles, spin_multipoles, node_features):
        q, m = electron_multipoles, spin_multipoles
        if (
            q.ndim != 2
            or q.shape[1] != 4
            or m.shape != q.shape
            or node_features.shape != (q.shape[0], self.node_irreps.dim)
        ):
            raise ValueError(
                "Expected electronic/spin multipoles [N,4] and matching node features"
            )

        # Rationalized sqrt(d.d + epsilon^2) - epsilon: smooth at zero.
        def norm(d):
            square = d.square().sum(-1, keepdim=True)
            return square / (
                torch.sqrt(square + self.norm_epsilon**2) + self.norm_epsilon
            )

        features = [q[:, :1], norm(q[:, 1:]), m[:, :1].square(), norm(m[:, 1:])]
        odd = []
        for sl, mul, ir in self.matches:
            h = node_features[:, sl].reshape(-1, mul, ir.dim)
            a, b = (q[:, :1], m[:, :1]) if ir.l == 0 else (q[:, 1:], m[:, 1:])
            features.append((a[:, None, :] * h).sum(-1))
            odd.append((b[:, None, :] * h).sum(-1))
        if odd:
            features.append(self.spin_invariant_mix(torch.cat(odd, -1)).square())
        return torch.cat(features, -1)

    def forward(self, electron_multipoles, spin_multipoles, node_features):
        return self.readout(
            self.invariants(electron_multipoles, spin_multipoles, node_features)
        ).squeeze(-1)


def assemble_matrix(
    edge_blocks: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    onsite_blocks: torch.Tensor,
) -> torch.Tensor:
    """Scatter a full directed edge list, symmetrize, and insert on-site blocks.

    edge_blocks: [E, B, B]; edge_index: [2, E] int64, on the same device.
    Every edge must have its reverse. Periodic self-image edges are allowed.
    Reverse-edge completeness is an input contract, not checked here.
    Repeated entries sum, so unintended duplicate edges must be removed upstream.
    onsite_blocks: required [N, B, B].
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
    if onsite_blocks.shape != (num_nodes, B, B):
        raise ValueError("Expected on-site blocks with shape [N, B, B]")
    D = 0.5 * (onsite_blocks + onsite_blocks.transpose(-1, -2))
    # Add, rather than overwrite, periodic self-image contributions.
    indices = torch.arange(num_nodes, device=H.device)[:, None] * B
    orbitals = indices + torch.arange(B, device=H.device)
    H[orbitals[:, :, None], orbitals[:, None, :]] += D
    return H


class MatrixConstruction(nn.Module):
    """Independent edge/on-site readouts followed by symmetric assembly."""

    def __init__(self, edge_irreps, n_s, n_p, onsite_irreps):
        super().__init__()
        self.edge_readout = SPBlockReadout(edge_irreps, n_s, n_p)
        self.onsite_readout = SPBlockReadout(onsite_irreps, n_s, n_p)

    def forward(self, edge_features, edge_index, num_nodes, onsite_features):
        return assemble_matrix(
            self.edge_readout(edge_features),
            edge_index,
            num_nodes,
            self.onsite_readout(onsite_features),
        )


def _uvu_product(left, right, targets, radial=False):
    """Keep one output copy per angular path, preserving left multiplicity."""
    left, right, targets = o3.Irreps(left), o3.Irreps(right), o3.Irreps(targets)
    allowed = {ir for mul, ir in targets if mul}
    outputs, instructions = [], []
    for i, (mul, ir_left) in enumerate(left):
        for j, (_, ir_right) in enumerate(right):
            for ir in ir_left * ir_right:
                if ir in allowed:
                    instructions.append((i, j, len(outputs), "uvu", True))
                    outputs.append((mul, ir))
    missing = allowed - {ir for _, ir in outputs}
    if missing:
        raise ValueError(f"Tensor product cannot supply required irreps: {missing}")
    ordered = o3.Irreps(outputs).sort()
    instructions = [
        (i, j, ordered.p[k], mode, train) for i, j, k, mode, train in instructions
    ]
    return o3.TensorProduct(
        left,
        right,
        ordered.irreps,
        instructions,
        internal_weights=not radial,
        shared_weights=not radial,
    )


def _orbital_order(n_s, n_p):
    return torch.tensor(
        list(range(n_s))
        + [n_s + 3 * shell + j for shell in range(n_p) for j in (2, 0, 1)]
    )


class OnsiteHamiltonianHead(nn.Module):
    """Quadratic plus direct linear coefficients; symmetric Cartesian blocks."""

    def __init__(self, node_irreps, n_s, n_p, feature_multiplicity=64):
        super().__init__()
        if feature_multiplicity < 1:
            raise ValueError("feature_multiplicity must be positive")
        features = o3.Irreps(
            [(feature_multiplicity, ir) for ir in ("0e", "1o", "1e", "2e")]
        )
        self.left = o3.Linear(node_irreps, node_irreps)
        self.right = o3.Linear(node_irreps, node_irreps)
        self.tp = _uvu_product(node_irreps, node_irreps, features)
        self.mix = o3.Linear(self.tp.irreps_out, features)
        self.readout = SPBlockReadout(features, n_s, n_p)
        self.linear = o3.Linear(
            node_irreps, self.readout.coefficient_irreps, biases=False
        )
        self.register_buffer("orbital_order", _orbital_order(n_s, n_p))

    def forward(self, features):
        quadratic = self.mix(self.tp(self.left(features), self.right(features)))
        coefficients = self.readout.readout(quadratic) + self.linear(features)
        blocks = self.readout.decode(coefficients)
        blocks = blocks.index_select(-2, self.orbital_order).index_select(
            -1, self.orbital_order
        )
        return (blocks + blocks.transpose(-1, -2)) / 2


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
        edge_mode="tensor_product",
        r_max=3.0,
        num_radial=8,
        feature_multiplicity=64,
        radial_hidden=32,
        sh_max_l=1,
    ):
        super().__init__()
        edge_mode = {
            "bilinear": "tensor_product",
            "linear_endpoints": "endpoint_sum",
        }.get(edge_mode, edge_mode)
        if edge_mode not in {"tensor_product", "endpoint_sum"}:
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
        if isinstance(sh_max_l, bool) or not isinstance(sh_max_l, int) or sh_max_l < 0:
            raise ValueError("sh_max_l must be a nonnegative integer")
        self.sh_irreps = o3.Irreps.spherical_harmonics(sh_max_l)
        self.r_max = r_max
        self.register_buffer("radial_centres", torch.linspace(0, r_max, num_radial))

        if edge_mode == "tensor_product":
            self.pair_tp = _uvu_product(
                self.node_irreps, self.node_irreps, self.node_irreps
            )
            self.pair_linear = o3.Linear(self.pair_tp.irreps_out, self.node_irreps)
        else:
            self.sender_linear = o3.Linear(self.node_irreps, self.node_irreps)
            self.receiver_linear = o3.Linear(self.node_irreps, self.node_irreps)

        self.angular_tp = _uvu_product(
            self.node_irreps, self.sh_irreps, self.feature_irreps, radial=True
        )
        self.angular_mix = o3.Linear(self.angular_tp.irreps_out, self.feature_irreps)
        self.radial_mlp = nn.Sequential(
            nn.Linear(num_radial, radial_hidden),
            nn.SiLU(),
            nn.Linear(radial_hidden, self.angular_tp.weight_numel),
        )

        self.onsite_head = OnsiteHamiltonianHead(
            node_irreps, n_s, n_p, feature_multiplicity
        )
        self.edge_readout = SPBlockReadout(self.feature_irreps, n_s, n_p)
        self.block_size = n_s + 3 * n_p
        self.register_buffer("orbital_order", _orbital_order(n_s, n_p))

    def forward(self, node_features, positions, edge_index, shifts=None, ptr=None):
        sender, receiver = edge_index
        vectors = positions[receiver] - positions[sender]
        if shifts is not None:
            vectors = vectors + shifts
        distances = vectors.norm(dim=-1)
        if torch.any(distances == 0):
            raise ValueError("Zero-length Hamiltonian edges are not allowed")
        # Reuse the MACE graph, but decode only edges within the Hamiltonian
        # support. The quintic envelope and its first two derivatives vanish
        # at this boundary, so removing the other edges preserves smoothness.
        within = distances < self.r_max
        edge_index = edge_index[:, within]
        sender, receiver = edge_index
        vectors, distances = vectors[within], distances[within]
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
        if self.edge_mode == "tensor_product":
            pair = self.pair_linear(
                self.pair_tp(node_features[sender], node_features[receiver])
            )
        else:
            left = self.sender_linear(node_features)
            right = self.receiver_linear(node_features)
            pair = left[sender] + right[receiver]
        edge_features = self.angular_mix(
            self.angular_tp(pair, sh, self.radial_mlp(radial))
        )
        edge_features = edge_features * cutoff[:, None]
        n = positions.shape[0]
        if ptr is None:
            ptr = torch.tensor([0, n], device=positions.device)
        blocks = self.edge_readout(edge_features)
        blocks = blocks.index_select(-2, self.orbital_order).index_select(
            -1, self.orbital_order
        )
        onsite = self.onsite_head(node_features)
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
            matrices.append(raw)
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
        from .occupation_root import bisect_chemical_potential, compiled_bisection

        root = compiled_bisection() if eps.is_cuda else bisect_chemical_potential
        mu = root(eps, count, tau)
    for _ in range(3):
        f = torch.sigmoid((mu - eps) / tau)
        slope = (f * (1 - f)).sum() / tau
        safe = slope.clamp_min(torch.finfo(eps.dtype).tiny)
        mu = mu + torch.where(slope > 0, (count - f.sum()) / safe, torch.zeros_like(mu))
    return torch.sigmoid((mu - eps) / tau), mu


def _occupations_and_entropy(eps, count, tau, tolerance):
    """One canonical channel; entropy is dimensionless negative entropy."""
    f, mu = _fill(eps, count, tau, tolerance)
    entropy = tau * 0
    if float(tau) > 0 and bool(torch.isfinite(mu)):
        x = (mu - eps) / tau
        entropy = (
            f * torch.nn.functional.logsigmoid(x)
            + (1 - f) * torch.nn.functional.logsigmoid(-x)
        ).sum()
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
    """Fixed-population density VJP from cached eigenpairs."""
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
    return (response + response.T) / 2


def _occupation_response(U, f, tau, grad):
    if float(tau) == 0:
        return torch.zeros_like(U)
    fp = -f * (1 - f) / tau
    total = fp.sum()
    mean = (fp * grad).sum() / torch.where(total != 0, total, torch.ones_like(total))
    return (U * (fp * (grad - mean))) @ U.T


class _CanonicalState(torch.autograd.Function):
    """One spin channel with an explicit derivative-order contract."""

    @staticmethod
    def forward(
        ctx,
        H,
        count,
        tau,
        tolerance,
        derivative_mode,
        higher_order_backend="smooth",
        spectral_chunk_size=None,
    ):
        eps, U = torch.linalg.eigh(H)
        f, mu, entropy = _occupations_and_entropy(eps, count, tau, tolerance)
        P = (U * f) @ U.T
        free = (f * eps).sum() + tau * entropy
        ctx.save_for_backward(H, count, tau, eps, U, f, mu, P)
        ctx.higher_order_backend = higher_order_backend
        ctx.spectral_chunk_size = spectral_chunk_size
        ctx.input_count = len(ctx.needs_input_grad)
        ctx.tolerance, ctx.derivative_mode = tolerance, derivative_mode
        ctx.mark_non_differentiable(eps, mu, U)
        ctx.set_materialize_grads(False)
        return P, free, eps, f, mu, U

    @staticmethod
    def backward(ctx, grad_P, grad_free, unused_eps, grad_f, unused_mu, unused_U):
        H, count, tau, eps, U, f, mu, saved_P = ctx.saved_tensors
        if torch.is_grad_enabled():
            if ctx.derivative_mode != "second_order":
                raise RuntimeError(
                    "first_order electronic state does not support higher derivatives"
                )
            if grad_f is not None:
                raise RuntimeError("Occupation outputs support first derivatives only")
            if ctx.higher_order_backend in {"spectral", "spectral_compiled", "spectral_compiled_last"}:
                from .spectral_derivatives import SpectralBackward

                A = torch.zeros_like(H) if grad_P is None else grad_P
                c = H.new_zeros(()) if grad_free is None else grad_free
                response = SpectralBackward.apply(
                    H,
                    A,
                    c,
                    eps.detach(),
                    U.detach(),
                    f.detach(),
                    mu.detach(),
                    saved_P.detach(),
                    tau,
                    count,
                    ctx.spectral_chunk_size,
                    ("last" if ctx.higher_order_backend == "spectral_compiled_last"
                     else ctx.higher_order_backend == "spectral_compiled"),
                )
                return (response,) + (None,) * (ctx.input_count - 1)
            P, _ = _smooth_density(H, count, tau, eps, mu, f)
            response = (
                H * 0
                if grad_P is None
                else torch.autograd.grad(
                    P, H, grad_P, create_graph=True, retain_graph=True
                )[0]
            )
        else:
            P = saved_P
            response = (
                torch.zeros_like(H)
                if grad_P is None
                else _spectral_response(U, eps, f, tau, grad_P, ctx.tolerance)
            )
            if grad_f is not None:
                response = response + _occupation_response(U, f, tau, grad_f)
        if grad_free is not None:
            response = response + grad_free * P
        return (response,) + (None,) * (ctx.input_count - 1)


class _GaussianDOS(torch.autograd.Function):
    """First-order smooth spectral observables, sharing an existing eigensolve."""

    @staticmethod
    def forward(ctx, H, eps, U, grid, width):
        ctx.save_for_backward(eps, U, grid, width)
        ctx.set_materialize_grads(False)
        dos, cumulative = [], []
        for energies in grid.split(128):
            z = (energies[:, None] - eps[None, :]) / width
            phi = torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)
            dos.append(phi.sum(-1) / width)
            cumulative.append((0.5 * torch.erfc(-z / math.sqrt(2))).sum(-1))
        return torch.cat(dos), torch.cat(cumulative)

    @staticmethod
    def backward(ctx, grad_dos, grad_cumulative):
        if torch.is_grad_enabled():
            raise RuntimeError("DOS outputs support first derivatives only")
        eps, U, grid, width = ctx.saved_tensors
        v = torch.zeros_like(eps)
        for start in range(0, grid.numel(), 128):
            end = start + 128
            z = (grid[start:end, None] - eps[None, :]) / width
            phi = torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)
            if grad_dos is not None:
                v += (grad_dos[start:end, None] * z * phi / width.square()).sum(0)
            if grad_cumulative is not None:
                v -= (grad_cumulative[start:end, None] * phi / width).sum(0)
        response = (U * v) @ U.T
        return (response + response.T) / 2, None, None, None, None


class ElectronicState(nn.Module):
    """RKS/UKS dense states; populations and smearing are constant inputs.

    RKS accepts a list of matrices; UKS a list of (alpha, beta) pairs.
    Spin mode is fixed at construction. Derivative mode is explicit per call.
    """

    def __init__(
        self,
        block_size,
        spin_mode,
        elec_temp_units="kelvin",
        dos_grid=None,
        dos_width=0.2,
        degeneracy_tolerance=1e-9,
        higher_order_backend="smooth",
        spectral_chunk_size=None,
    ):
        super().__init__()
        if spin_mode not in {"RKS", "UKS"}:
            raise ValueError("spin_mode must be RKS or UKS")
        if elec_temp_units not in {"kelvin", "eV"}:
            raise ValueError("elec_temp_units must be kelvin or eV")
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        if not math.isfinite(degeneracy_tolerance) or degeneracy_tolerance < 0:
            raise ValueError("degeneracy_tolerance must be finite and nonnegative")
        grid = (
            torch.linspace(-15, 5, 100)
            if dos_grid is None
            else torch.as_tensor(dos_grid, dtype=torch.get_default_dtype())
        )
        width = torch.as_tensor(dos_width, dtype=torch.get_default_dtype())
        if grid.requires_grad or width.requires_grad:
            raise ValueError("DOS grid and width must be constant")
        if (
            grid.ndim != 1
            or grid.numel() == 0
            or not bool(torch.isfinite(grid).all())
            or not bool((grid[1:] > grid[:-1]).all())
        ):
            raise ValueError(
                "dos_grid must be finite, nonempty and strictly increasing"
            )
        if width.ndim != 0 or not bool(torch.isfinite(width)) or float(width) <= 0:
            raise ValueError("dos_width must be positive and finite")
        self.register_buffer("dos_grid", grid.clone())
        self.register_buffer("dos_width", width.clone())
        if higher_order_backend not in {"smooth", "spectral", "spectral_compiled", "spectral_compiled_last"}:
            raise ValueError("higher_order_backend must be smooth, spectral, spectral_compiled or spectral_compiled_last")
        if spectral_chunk_size is not None and (
            not isinstance(spectral_chunk_size, int) or spectral_chunk_size < 1
        ):
            raise ValueError("spectral_chunk_size must be a positive integer or None")
        self.higher_order_backend = higher_order_backend
        self.spectral_chunk_size = spectral_chunk_size
        self.block_size, self.spin_mode = block_size, spin_mode
        self.elec_temp_units, self.degeneracy_tolerance = (
            elec_temp_units,
            degeneracy_tolerance,
        )

    def thermal_energy(self, temperature):
        return temperature * (
            8.617333262145e-5 if self.elec_temp_units == "kelvin" else 1.0
        )

    def _channel(self, H, count, tau, mode, full, eigenvectors):
        if (
            H.ndim != 2
            or H.shape[0] != H.shape[1]
            or H.shape[0] == 0
            or H.shape[0] % self.block_size
        ):
            raise ValueError(
                "Hamiltonian must be nonempty, square and divisible by block_size"
            )
        if not bool(torch.isfinite(H).all()) or not torch.allclose(H, H.T):
            raise ValueError("Hamiltonian must be finite and symmetric")
        if mode == "none":
            with torch.no_grad():
                P, free, eps, f, mu, U = _CanonicalState.apply(
                    H.detach(), count, tau, self.degeneracy_tolerance, mode
                )
                dos, cumulative = _GaussianDOS.apply(
                    H.detach(), eps, U, self.dos_grid, self.dos_width
                )
        else:
            P, free, eps, f, mu, U = _CanonicalState.apply(
                H,
                count,
                tau,
                self.degeneracy_tolerance,
                mode,
                getattr(self, "higher_order_backend", "smooth"),
                getattr(self, "spectral_chunk_size", None),
            )
            dos, cumulative = _GaussianDOS.apply(
                H, eps, U, self.dos_grid, self.dos_width
            )
        band = (P * (H.detach() if mode == "none" else H)).sum()
        B = self.block_size
        onsite = (
            P.reshape(-1, B, H.shape[0] // B, B)
            .diagonal(dim1=0, dim2=2)
            .permute(2, 0, 1)
        )
        diagnostic = dict(
            eigenvalues=eps,
            occupations=f,
            chemical_potential=mu,
            count_residual=f.detach().sum() - count,
        )
        if full:
            diagnostic["density"] = P
        if eigenvectors:
            diagnostic["eigenvectors"] = U
        return onsite, band, free - band, dos, cumulative, diagnostic

    def forward(
        self,
        hamiltonians,
        N_alpha,
        N_beta,
        elec_temp,
        *,
        derivative_mode,
        return_full_density=False,
        return_eigenvectors=False,
    ):
        if derivative_mode not in {"none", "first_order", "second_order"}:
            raise ValueError("Unknown derivative_mode")
        if derivative_mode != "none" and not torch.is_grad_enabled():
            raise RuntimeError(
                "Differentiable electronic mode requires gradient recording"
            )
        ng = len(hamiltonians)
        if not ng:
            raise ValueError("Require at least one Hamiltonian")
        for value in (N_alpha, N_beta, elec_temp):
            if value.requires_grad:
                raise ValueError("Populations and temperature must be constant inputs")
            if value.shape != (ng,) or not bool(torch.isfinite(value).all()):
                raise ValueError(
                    "Populations and temperature must be finite [num_graphs] tensors"
                )
        if bool((elec_temp < 0).any()):
            raise ValueError("Electronic temperature must be nonnegative")
        if derivative_mode == "second_order" and bool((elec_temp == 0).any()):
            raise ValueError("second_order requires positive electronic smearing")
        if self.spin_mode == "RKS" and not torch.equal(N_alpha, N_beta):
            raise ValueError("RKS requires N_alpha == N_beta")
        if self.spin_mode == "UKS" and bool((N_alpha < N_beta).any()):
            raise ValueError("UKS requires N_alpha >= N_beta")
        gamma, spin, bands, entropies, dos, cumulative, states = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        mus, residuals = [], []
        for g, matrices in enumerate(hamiltonians):
            if self.spin_mode == "RKS":
                if not isinstance(matrices, torch.Tensor):
                    raise ValueError("RKS accepts one matrix per graph")
                matrices = (matrices,)
            elif not isinstance(matrices, (tuple, list)) or len(matrices) != 2:
                raise ValueError("UKS accepts an alpha/beta matrix pair per graph")
            if self.spin_mode == "UKS" and matrices[0].shape != matrices[1].shape:
                raise ValueError("UKS matrices must share an orbital layout")
            tau = self.thermal_energy(elec_temp[g])
            channels = [
                self._channel(
                    H,
                    (N_alpha, N_beta)[s][g],
                    tau,
                    derivative_mode,
                    return_full_density,
                    return_eigenvectors,
                )
                for s, H in enumerate(matrices)
            ]
            a, b = channels[0], channels[-1]
            gamma.append(a[0] + b[0])
            spin.append(a[0] - b[0])
            bands.append(a[1] + b[1])
            entropies.append(a[2] + b[2])
            dos.append(torch.stack((a[3], b[3])))
            cumulative.append(torch.stack((a[4], b[4])))
            states.append(tuple(c[5] for c in channels))
            mus.append(
                torch.stack((a[5]["chemical_potential"], b[5]["chemical_potential"]))
            )
            residuals.append(
                torch.stack((a[5]["count_residual"], b[5]["count_residual"]))
            )
        return dict(
            gamma=torch.cat(gamma),
            gamma_spin=torch.cat(spin),
            band_energy=torch.stack(bands),
            entropy_energy=torch.stack(entropies),
            dos=torch.stack(dos),
            cumulative_dos=torch.stack(cumulative),
            spectral_states=states,
            chemical_potentials=torch.stack(mus),
            count_residuals=torch.stack(residuals),
            derivative_mode=derivative_mode,
        )
