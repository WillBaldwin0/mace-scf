"""Numerical contracts for the first non-SCF model (CPU, double precision)."""

import pytest
import torch
from e3nn import o3
from mace_scf.electrostatics.mldftb import OneShotMLUKS, OneShotMLRKS
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
        total_charge=torch.tensor([0.0]),
        N_alpha=torch.tensor([1.7]),
        N_beta=torch.tensor([1.3]),
        elec_temp=torch.tensor([0.2]),
        cell=(torch.eye(3) * 6)[None],
        pbc=torch.full((1, 3), periodic),
        batch=torch.zeros(3, dtype=torch.long),
        ptr=torch.tensor([0, 3]),
    )


def model(mode="bilinear"):
    return OneShotMLUKS(
        [1],
        [1.0],
        hidden_irreps="2x0e+2x1o",
        num_interactions=1,
        elec_temp_units="eV",
        matrix_feature_multiplicity=2,
        edge_mode=mode,
    )


@pytest.mark.parametrize("mode", ["bilinear", "linear_endpoints"])
@pytest.mark.parametrize("periodic", [False, True])
def test_model_forces_and_training(mode, periodic):
    m = model(mode)
    d = graph(periodic)
    out = m(d, training=True, compute_stress=periodic, return_electronic_state=True)
    assert out["count_residuals"].abs().max() < 1e-12
    torch.testing.assert_close(out["total_charge"], torch.zeros(1), atol=1e-12, rtol=0)
    for h in out["hamiltonians"][0]:
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
    p = graph(False)
    both = {
        k: torch.cat((d[k], p[k]))
        for k in [
            "positions",
            "node_attrs",
            "total_charge",
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
        DensityElectrostatics(1, 1, [1.0], pp_scalar="S", kspace_cutoff=5.0)
    with pytest.raises(NotImplementedError):
        DensityElectrostatics(1, 1, [1.0], keep_quadrupoles=True, kspace_cutoff=5.0)
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
        total_charge=torch.tensor([0.0]),
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


@pytest.mark.parametrize("mode", ["bilinear", "linear_endpoints"])
def test_hamiltonian_cutoff_support_and_smoothness(mode):
    builder = HamiltonianBuilder("2x0e+2x1o", n_s=1, n_p=1, r_max=2.0, edge_mode=mode)
    features = torch.randn(2, 8)
    edges = torch.tensor([[0, 1], [1, 0]])

    def block(distance):
        positions = torch.stack(
            (torch.zeros(3), torch.stack((distance, distance * 0, distance * 0)))
        )
        return builder(features, positions, edges)[0][:4, 4:]

    assert block(torch.tensor(1.0)).norm() > 0
    for r in [2.0, 2.1]:
        assert torch.count_nonzero(block(torch.tensor(r))) == 0
    norms = []
    for offset in [1e-3, 1e-4]:
        r = torch.tensor(2.0 - offset, requires_grad=True)
        value = block(r).sum()
        first = torch.autograd.grad(value, r, create_graph=True)[0]
        second = torch.autograd.grad(first, r)[0]
        norms.append(torch.stack([value.abs(), first.abs(), second.abs()]))
    assert torch.all(norms[1] < norms[0] * 0.2)
    # Periodic shifts, rather than unshifted atom separations, set the mask.
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    shifts = torch.tensor([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    assert (
        torch.count_nonzero(builder(features, positions, edges, shifts)[0][:4, 4:]) == 0
    )


@pytest.mark.parametrize("cutoff", [0.0, -1.0, 6.0, float("nan")])
def test_invalid_hamiltonian_cutoff(cutoff):
    with pytest.raises(ValueError, match="hamiltonian_cutoff"):
        OneShotMLUKS([1], [1.0], r_max=5.0, hamiltonian_cutoff=cutoff)


def test_short_hamiltonian_cutoff_model_force_training():
    m = OneShotMLUKS(
        [1],
        [1.0],
        r_max=5.0,
        hamiltonian_cutoff=0.5,
        hidden_irreps="2x0e+2x1o",
        num_interactions=1,
        elec_temp_units="eV",
        matrix_feature_multiplicity=2,
    )
    assert float(m.r_max) == 5.0
    assert m.hamiltonian.r_max == 0.5
    out = m(graph(), training=True)
    (out["energy"].square().sum() + out["forces"].square().sum()).backward()
    assert all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )


@pytest.mark.parametrize("kind", ["internal", "free"])
def test_rks_uks_equivalence_and_combined_spectral_force_loss(kind):
    kwargs = dict(
        hidden_irreps="2x0e+2x1o",
        num_interactions=1,
        matrix_feature_multiplicity=3,
        elec_temp_units="eV",
        energy_kind=kind,
    )
    rks = OneShotMLRKS([1], [1.0], **kwargs)
    uks = OneShotMLUKS([1], [1.0], **kwargs)
    uks.load_state_dict(rks.state_dict(), strict=False)
    assert any(torch.count_nonzero(p) for p in uks.exchange_head.parameters())
    for p in uks.exchange_head.parameters():
        p.data.zero_()
    d = graph()
    d["N_alpha"] = d["N_beta"] = torch.tensor([1.5])
    r = rks(d, training=True, return_electronic_state=True)
    u = uks(d, training=True, return_electronic_state=True)
    for key in ("energy", "forces", "gamma", "gamma_spin", "dos", "cumulative_dos"):
        torch.testing.assert_close(r[key], u[key])
    assert r["derivative_mode"] == u["derivative_mode"] == "second_order"
    for model_, out in ((rks, r), (uks, u)):
        (
            out["energy"].sum()
            + out["forces"].square().sum()
            + out["dos"].square().sum() * 0.01
            + out["cumulative_dos"].sum() * 0.01
        ).backward()
        assert all(
            torch.isfinite(p.grad).all()
            for p in model_.parameters()
            if p.grad is not None
        )
    for name, p in rks.named_parameters():
        q = dict(uks.named_parameters())[name]
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-8, rtol=1e-6)
    for training, forces, mode in (
        (False, False, "none"),
        (False, True, "first_order"),
        (True, False, "first_order"),
    ):
        out = rks(
            d, training=training, compute_force=forces, return_electronic_state=True
        )
        assert out["derivative_mode"] == mode


def test_exchange_is_onsite_and_in_band_energy():
    m = model()
    out = m(graph(), training=True, compute_force=False, return_electronic_state=True)
    ha, hb = out["hamiltonians"][0]
    delta = (ha - hb) / 2
    for i in range(3):
        for j in range(3):
            if i != j:
                torch.testing.assert_close(
                    delta[4 * i : 4 * i + 4, 4 * j : 4 * j + 4], torch.zeros(4, 4)
                )
    assert delta.norm() > 0
    pa, pb = (s["density"] for s in out["spectral_states"][0])
    expected = ((pa + pb) * (ha + hb) / 2).sum() + ((pa - pb) * delta).sum()
    torch.testing.assert_close(expected, out["band_energy"][0])
    out["energy"].sum().backward()
    assert any(
        p.grad is not None and p.grad.norm() > 0 for p in m.exchange_head.parameters()
    )
    with pytest.raises(ValueError, match="N_alpha >= N_beta"):
        m({**graph(), "N_alpha": torch.tensor([1.0]), "N_beta": torch.tensor([2.0])})


@pytest.mark.parametrize("edge_mode", ["tensor_product", "endpoint_sum"])
def test_matrix_equivariance_and_configurable_harmonics(edge_mode):
    irreps = o3.Irreps("3x0e+3x1o")
    builder = HamiltonianBuilder(
        irreps, n_s=2, n_p=2, feature_multiplicity=5, sh_max_l=2, edge_mode=edge_mode
    )
    x = torch.randn(3, 3)
    features = torch.randn(3, irreps.dim)
    edges = graph()["edge_index"]
    R = -o3.rand_matrix()  # include reflections
    perm = torch.tensor([1, 2, 0])
    Q = R[perm][:, perm]
    rotated = features @ irreps.D_from_matrix(Q).T
    H = builder(features, x, edges)[0]
    HR = builder(rotated, x @ R.T, edges)[0]
    atomD = torch.block_diag(torch.eye(2), R, R)
    D = torch.block_diag(atomD, atomD, atomD)
    torch.testing.assert_close(HR, D @ H @ D.T, atol=1e-8, rtol=1e-7)
    torch.testing.assert_close(H, H.T)
    with pytest.raises(ValueError, match="required irreps"):
        HamiltonianBuilder(irreps, sh_max_l=0)
