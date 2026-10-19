"""Correctness checks shared by tests and the spectral-backward benchmark."""

import torch
from mace_scf.electrostatics.matrix_ops import _CanonicalState
from mace_scf.electrostatics.mldftb import OneShotMLRKS, OneShotMLUKS
from mace_scf.utils.model_training_wrappers import make_model_wrapper


def objective(H, count, tau, backend, A, chunk=None):
    P, free = _CanonicalState.apply(
        (H + H.T) / 2,
        H.new_tensor(count),
        H.new_tensor(tau),
        1e-9,
        "second_order",
        backend,
        chunk,
    )[:2]
    return (P * A).sum() + 0.1 * P.square().sum() + 0.01 * free.square()


def matrix_check(H, count, tau, chunk=None, backend="spectral"):
    H = H.detach().requires_grad_()
    A = torch.randn_like(H)
    Z = torch.randn_like(H)
    Z = (Z + Z.T) / 2
    Z = Z / Z.norm()
    values = []
    for backend in ("smooth", backend):
        loss = objective(H, count, tau, backend, A, chunk)
        g = torch.autograd.grad(loss, H, create_graph=True)[0]
        h = torch.autograd.grad((g * Z).sum(), H)[0]
        values.append((g.detach(), h.detach()))
    torch.testing.assert_close(values[1][0], values[0][0], atol=2e-8, rtol=2e-5)
    torch.testing.assert_close(values[1][1], values[0][1], atol=2e-8, rtol=2e-5)
    errors = []
    for step in (1e-4, 3e-5, 1e-5):
        grads = []
        for sign in (-1, 1):
            h = (H.detach() + sign * step * Z).requires_grad_()
            loss = objective(h, count, tau, "spectral", A, chunk)
            grads.append(torch.autograd.grad(loss, h)[0])
        numerical = (grads[1] - grads[0]) / (2 * step)
        errors.append(float((numerical - values[1][1]).abs().max()))
    if min(errors) > 2e-6 + 2e-4 * float(values[1][1].abs().max()):
        raise AssertionError(f"Matrix finite difference failed: {errors}")
    return dict(
        first_vs_smooth_max=float((values[0][0] - values[1][0]).abs().max()),
        second_vs_smooth_max=float((values[0][1] - values[1][1]).abs().max()),
        finite_difference_steps=[1e-4, 3e-5, 1e-5],
        finite_difference_max_errors=errors,
    )


def wrapper_check(
    device="cpu", spin="RKS", periodic=False, batch_size=1, chunk=None, force_only=False, backend="spectral"
):
    """Force-loss parameter directional FD through the actual training wrapper."""
    from ase import Atoms
    from mace.data import KeySpecification, config_from_atoms
    from mace.tools import AtomicNumberTable, torch_geometric
    from mace_scf.data import ExtAtomicData

    cls = OneShotMLRKS if spin == "RKS" else OneShotMLUKS
    model = cls(
        [1],
        [1.0],
        hidden_irreps="2x0e+2x1o",
        num_interactions=1,
        matrix_feature_multiplicity=2,
        elec_temp_units="eV",
        xc_type="onsite",
        spectral_chunk_size=chunk,
    ).to(device)
    keys = KeySpecification(
        info_keys={k: k for k in ("N_alpha", "N_beta", "total_charge", "elec_temp")},
        arrays_keys={},
    )
    graphs = []
    for i in range(batch_size):
        a = Atoms(
            "H3",
            positions=[[0, 0, 0], [1.1 + 0.02 * i, 0.2, 0], [0.1, 1.2, 0.3]],
            cell=[5, 5, 5],
            pbc=periodic,
        )
        a.info.update(
            N_alpha=1.5 if spin == "RKS" else 1.7,
            N_beta=1.5 if spin == "RKS" else 1.3,
            total_charge=0.0,
            elec_temp=0.3,
        )
        graphs.append(
            ExtAtomicData.from_config(
                config_from_atoms(a, key_specification=keys),
                AtomicNumberTable([1]),
                float(model.r_max),
                preserve_cell=True,
            )
        )
    data = (
        next(iter(torch_geometric.dataloader.DataLoader(graphs, batch_size=batch_size)))
        .to(device)
        .to_dict()
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    wrapper = make_model_wrapper(
        model, optimizer, dict(forces=True, virials=False, stress=False)
    )
    parameters = list(model.parameters())
    directions = [torch.randn_like(p) for p in parameters]
    norm = torch.sqrt(sum(d.square().sum() for d in directions))
    directions = [d / norm for d in directions]
    base = [p.detach().clone() for p in parameters]

    def loss(training):
        fresh = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()
        }
        out = wrapper(model, fresh, training=training)
        force_loss = out["forces"].square().mean()
        return (
            force_loss
            if force_only
            else 0.1 * out["energy"].square().mean() + force_loss
        )

    derivatives = []
    gradients = []
    for backend in ("smooth", backend):
        model.electronic_state.higher_order_backend = backend
        optimizer.zero_grad(set_to_none=True)
        loss(True).backward()
        grads = [
            p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            for p in parameters
        ]
        gradients.append(grads)
        derivatives.append(float(sum((g * d).sum() for g, d in zip(grads, directions))))
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-4)
    errors = []
    differences = []
    try:
        for step in (1e-4, 3e-5, 1e-5):
            losses = []
            for sign in (-1, 1):
                with torch.no_grad():
                    for p, original, d in zip(parameters, base, directions):
                        p.copy_(original + sign * step * d)
                losses.append(float(loss(False).detach()))
            fd = (losses[1] - losses[0]) / (2 * step)
            differences.append(fd)
            errors.append(abs(fd - derivatives[1]))
    finally:
        with torch.no_grad():
            for p, original in zip(parameters, base):
                p.copy_(original)
    if min(errors) > 2e-6 + 2e-3 * abs(derivatives[1]):
        raise AssertionError(
            f"Wrapper finite difference failed: analytic={derivatives[1]}, FD={differences}"
        )
    return dict(
        force_only=force_only,
        spin=spin,
        periodic=periodic,
        batch_size=batch_size,
        smooth_directional=derivatives[0],
        spectral_directional=derivatives[1],
        parameter_gradient_max_error=max(
            float((a - b).abs().max()) for a, b in zip(*gradients)
        ),
        finite_difference_steps=[1e-4, 3e-5, 1e-5],
        finite_difference_values=differences,
        finite_difference_errors=errors,
    )
