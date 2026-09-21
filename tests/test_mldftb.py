"""Numerical contracts for the first non-SCF model (CPU, double precision)."""

import pytest
import torch
from e3nn import o3
from mace_scf.electrostatics.mldftb import MLDFTB
from mace_scf.electrostatics.matrix_ops import ElectronicState, HamiltonianBuilder
from mace_scf.electrostatics.density_electrostatics import DensityElectrostatics


@pytest.fixture(autouse=True)
def double_precision():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(14)
    yield
    torch.set_default_dtype(old)


def graph(periodic=False):
    return dict(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.3, 0.2], [0.2, 1.1, 0.4]]),
        node_attrs=torch.ones(3, 1),
        edge_index=torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]]),
        N_alpha=torch.tensor([1.7]),
        N_beta=torch.tensor([1.3]),
        elec_temp=torch.tensor([0.2]),
        cell=(torch.eye(3) * 6)[None],
        pbc=torch.full((1, 3), periodic),
        batch=torch.zeros(3, dtype=torch.long),
        ptr=torch.tensor([0, 3]),
    )


def model(mode="bilinear"):
    return MLDFTB(
        [1],
        [1.0],
        hidden_irreps="2x0e+2x1o",
        num_interactions=1,
        elec_temp_units="eV",
        matrix_feature_multiplicity=2,
        edge_mode=mode,
    )


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("policy", ["full", "subtract_nuclear", "subtract_atomic"])
def test_explicit_potential(periodic, policy):
    d = graph(periodic)
    es = DensityElectrostatics(1, 1, [1.0], self_policy=policy)
    g = torch.randn(3, 4, 4, requires_grad=True)
    out = es(
        g,
        d,
        torch.zeros(3, dtype=torch.long),
        compute_potential=True,
        external_field=torch.tensor([[0.1, 0.2, 0.3]]),
    )
    deriv = torch.autograd.grad(out["electrostatic_energy"].sum(), g)[0]
    torch.testing.assert_close(deriv, out["effective_potential"], atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(
        out["atomic_electron_counts"], g.diagonal(dim1=-2, dim2=-1).sum(-1)
    )


def test_electronic_derivatives():
    state = ElectronicState(4, "eV")
    h = torch.randn(4, 4, requires_grad=True)

    def fn(h):
        return state(
            [(h + h.T) / 2],
            torch.tensor([1.3]),
            torch.tensor([0.8]),
            torch.tensor([0.3]),
        )["gamma"]

    assert torch.autograd.gradcheck(fn, (h,))
    assert torch.autograd.gradgradcheck(fn, (h,))
    h = torch.eye(4, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (h,))
    assert torch.autograd.gradgradcheck(fn, (h,))


@pytest.mark.parametrize("mode", ["bilinear", "linear_endpoints"])
@pytest.mark.parametrize("periodic", [False, True])
def test_model_forces_and_training(mode, periodic):
    m = model(mode)
    d = graph(periodic)
    out = m(d, training=True, compute_stress=periodic, return_electronic_state=True)
    assert out["count_residuals"].abs().max() < 1e-12
    torch.testing.assert_close(out["total_charge"], torch.zeros(1), atol=1e-12, rtol=0)
    h = out["hamiltonians"][0]
    torch.testing.assert_close(h, h.T)
    step = 1e-5
    plus = {**d, "positions": d["positions"].clone()}
    plus["positions"][1, 0] += step
    minus = {**d, "positions": d["positions"].clone()}
    minus["positions"][1, 0] -= step
    numerical = -(
        m(plus, compute_force=False)["energy"] - m(minus, compute_force=False)["energy"]
    ) / (2 * step)
    torch.testing.assert_close(out["forces"][1, 0], numerical[0], atol=2e-7, rtol=2e-5)
    (out["energy"].sum() + out["forces"].square().sum()).backward()
    assert all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )
    if periodic:
        strain = torch.diag(torch.tensor([step, 0.0, 0.0]))
        ep = m(
            {
                **d,
                "positions": d["positions"] @ (torch.eye(3) + strain),
                "cell": d["cell"] @ (torch.eye(3) + strain),
            },
            compute_force=False,
        )["energy"]
        em = m(
            {
                **d,
                "positions": d["positions"] @ (torch.eye(3) - strain),
                "cell": d["cell"] @ (torch.eye(3) - strain),
            },
            compute_force=False,
        )["energy"]
        torch.testing.assert_close(
            out["stress"][0, 0, 0],
            ((ep - em) / (2 * step * 216))[0],
            atol=1e-8,
            rtol=2e-4,
        )


def test_rotation_and_batch():
    m = model()
    d = graph()
    out = m(d)
    R = o3.rand_matrix()
    rot = m({**d, "positions": d["positions"] @ R.T})
    # Open dipoles are evaluated with the library's finite-difference stencil.
    torch.testing.assert_close(rot["energy"], out["energy"], atol=2e-6, rtol=1e-4)
    torch.testing.assert_close(rot["dipole"], out["dipole"] @ R.T, atol=1e-9, rtol=1e-7)
    p = graph(True)
    both = {
        k: torch.cat((d[k], p[k]))
        for k in [
            "positions",
            "node_attrs",
            "N_alpha",
            "N_beta",
            "elec_temp",
            "cell",
            "pbc",
        ]
    }
    both.update(
        edge_index=torch.cat((d["edge_index"], p["edge_index"] + 3), 1),
        batch=torch.tensor([0, 0, 0, 1, 1, 1]),
        ptr=torch.tensor([0, 3, 6]),
    )
    bat = m(both)
    torch.testing.assert_close(
        bat["energy"], torch.cat((out["energy"], m(p)["energy"]))
    )


def test_options_fail_explicitly():
    with pytest.raises(NotImplementedError):
        DensityElectrostatics(1, 1, [1.0], pp_scalar="S")
    with pytest.raises(NotImplementedError):
        DensityElectrostatics(1, 1, [1.0], keep_quadrupoles=True)
    with pytest.raises(ValueError, match="inconsistent"):
        model()({**graph(), "total_charge": torch.tensor([1.0])})


def test_periodic_self_images_and_isolated_atom():
    from mace_scf.electrostatics.matrix_ops import assemble_matrix

    edge = torch.randn(2, 4, 4)
    onsite = torch.randn(1, 4, 4)
    H = assemble_matrix(edge, torch.zeros(2, 2, dtype=torch.long), 1, onsite)
    raw = edge.sum(0) + onsite[0]
    torch.testing.assert_close(H, (raw + raw.T) / 2)
    m = model()
    d = dict(
        positions=torch.zeros(1, 3),
        node_attrs=torch.ones(1, 1),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        N_alpha=torch.tensor([0.6]),
        N_beta=torch.tensor([0.4]),
        elec_temp=torch.tensor([0.2]),
    )
    for kind in ("internal", "free"):
        m.energy_kind = kind
        out = m(d, training=True)
        (out["energy"].sum() + out["forces"].square().sum()).backward()
        assert all(
            torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
        )
        torch.testing.assert_close(out["forces"], torch.zeros(1, 3))
        m.zero_grad()
    d.update(
        edge_index=torch.zeros(2, 2, dtype=torch.long),
        unit_shifts=torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        cell=(torch.eye(3) * 3)[None],
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )
    out = m(d, compute_stress=True, training=True)
    out["forces"].square().sum().backward()
    assert torch.isfinite(out["stress"]).all()


def test_data_fields():
    from ase import Atoms
    from mace.data import config_from_atoms, KeySpecification
    from mace.tools import AtomicNumberTable, torch_geometric
    from mace_scf.data.new_atomic_data import ExtAtomicData, update_keyspec_from_kwargs

    keys = update_keyspec_from_kwargs(
        KeySpecification(),
        dict(
            N_alpha_key="na",
            N_beta_key="nb",
            elec_temp_key="temperature",
            effective_nuclear_charges_key="zeff",
        ),
    )
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms.info.update(na=1.0, nb=1.0, temperature=300.0)
    import numpy as np

    atoms.arrays["zeff"] = np.ones(2)
    config = config_from_atoms(atoms, key_specification=keys)
    d = ExtAtomicData.from_config(config, AtomicNumberTable([1]), 3.0)
    batch = next(iter(torch_geometric.dataloader.DataLoader([d, d], batch_size=2)))
    torch.testing.assert_close(batch.N_alpha, torch.ones(2))
    torch.testing.assert_close(batch.N_beta, torch.ones(2))
    torch.testing.assert_close(batch.elec_temp, torch.full((2,), 300.0))
    torch.testing.assert_close(batch.effective_nuclear_charges, torch.ones(4))


@pytest.mark.parametrize("mode", ["molecule_in_box", "slab"])
def test_reciprocal_boundary_projections(mode):
    d = graph(False)
    if mode == "slab":
        d["pbc"] = torch.tensor([[True, True, False]])
    es = DensityElectrostatics(1, 1, [1.0], pbc_handling=mode)
    g = torch.randn(3, 4, 4, requires_grad=True)
    out = es(g, d, torch.zeros(3, dtype=torch.long), compute_potential=True)
    deriv = torch.autograd.grad(out["electrostatic_energy"].sum(), g)[0]
    torch.testing.assert_close(deriv, out["effective_potential"], atol=2e-9, rtol=2e-9)


def test_free_energy_degenerate_derivatives():
    from mace_scf.electrostatics.matrix_ops import _CanonicalState

    h = torch.eye(4, requires_grad=True)

    def fn(h):
        return _CanonicalState.apply(
            (h + h.T) / 2, torch.tensor([1.3, 0.8]), torch.tensor(0.2), 1e-9
        )[1]

    assert torch.autograd.gradcheck(fn, (h,))
    assert torch.autograd.gradgradcheck(fn, (h,))


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("counts", [[1.3, 1.3], [1.3, 0.8]])
def test_shared_spectral_solve(monkeypatch, training, counts):
    import mace_scf.electrostatics.matrix_ops as ops

    calls = {"eigh": 0, "eigvalsh": 0, "smooth": 0}
    for name in ("eigh", "eigvalsh"):
        original = getattr(torch.linalg, name)

        def wrapper(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch.linalg, name, wrapper)
    smooth = ops._smooth_density

    def smooth_wrapper(*args):
        calls["smooth"] += 1
        return smooth(*args)

    monkeypatch.setattr(ops, "_smooth_density", smooth_wrapper)
    h = torch.randn(4, 4, requires_grad=True)
    P, free, *_ = ops._CanonicalState.apply(
        (h + h.T) / 2, torch.tensor(counts), torch.tensor(0.3), 1e-9
    )
    energy = free + (P * torch.randn_like(P)).sum().square()
    force = torch.autograd.grad(energy, h, create_graph=training)[0]
    if training:
        force.square().sum().backward()
        assert torch.isfinite(h.grad).all()
    assert calls["eigh"] == 1
    assert calls["eigvalsh"] == 0
    if training:
        assert calls["smooth"] == (1 if counts[0] == counts[1] else 2)
    else:
        assert calls["smooth"] == 0


@pytest.mark.parametrize("counts", [[2.0, 2.0], [0.0, 4.0]])
def test_insulating_state_derivatives(counts):
    from mace_scf.electrostatics.matrix_ops import _CanonicalState

    q, _ = torch.linalg.qr(torch.randn(4, 4))
    h = (q @ torch.diag(torch.tensor([-3.0, -2.0, 2.0, 3.0])) @ q.T).requires_grad_()

    def fn(h):
        P, free, *_ = _CanonicalState.apply(
            (h + h.T) / 2, torch.tensor(counts), torch.tensor(0.01), 1e-9
        )
        return P, free

    assert torch.autograd.gradcheck(fn, (h,))
    assert torch.autograd.gradgradcheck(fn, (h,))
    if counts[0] == 2:
        probe = q[:, 0, None] @ q[:, 3, None].T
        density = fn(h)[0][0]
        response = torch.autograd.grad((density * probe).sum(), h)[0]
        assert response.norm() > 0.01


def test_canonical_state_count_width_derivatives():
    from mace_scf.electrostatics.matrix_ops import _CanonicalState

    h = torch.randn(3, 3, requires_grad=True)
    counts = torch.tensor([1.2, 1.2], requires_grad=True)
    tau = torch.tensor(0.4, requires_grad=True)

    def fn(h, counts, tau):
        return _CanonicalState.apply((h + h.T) / 2, counts, tau, 1e-9)[:2]

    assert torch.autograd.gradcheck(fn, (h, counts, tau))
    assert torch.autograd.gradgradcheck(fn, (h, counts, tau))


def test_insulating_reconstruction_rejects_wrong_count():
    from mace_scf.electrostatics.matrix_ops import _smooth_density

    h = torch.diag(torch.tensor([-3.0, -2.0, 2.0, 3.0]))
    with pytest.raises(RuntimeError, match="incorrect electron count"):
        _smooth_density(
            h,
            torch.tensor(1.0),
            torch.tensor(0.01),
            h.diagonal(),
            torch.tensor(0.0),
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )


def test_zero_width_inference_and_force_gradient_error():
    from mace_scf.electrostatics.matrix_ops import _CanonicalState

    h = torch.diag(torch.tensor([-3.0, -2.0, 2.0, 3.0])).requires_grad_()
    P, free, *_ = _CanonicalState.apply(
        h, torch.tensor([2.0, 2.0]), torch.tensor(0.0), 1e-9
    )
    gradient = torch.autograd.grad(free + P.square().sum(), h, retain_graph=True)[0]
    assert torch.isfinite(gradient).all()
    with pytest.raises(RuntimeError, match="positive electronic smearing"):
        torch.autograd.grad(free, h, create_graph=True)
