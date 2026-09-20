"""Training registration, checkpoint/ASE interoperability and electronic caching."""

import os
from pathlib import Path
import subprocess
import sys

from ase import Atoms
from ase.io import write
import numpy as np
import pytest
import torch
import yaml

from mace.tools import AtomicNumberTable
from mace_scf.calculators.mldftb import MLDFTBCalculator
from mace_scf.electrostatics import MLDFTB
from mace_scf.utils import extended_arg_parser
from mace_scf.utils.check_args import check_config_conflicts
from mace_scf.utils.run_train_utils import build_model, get_param_options
from mace_scf.utils.model_training_wrappers import make_model_wrapper

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def precision():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2)
    yield
    torch.set_default_dtype(old)


def atoms(periodic=False):
    a = Atoms(
        "H3",
        positions=[[0, 0, 0], [1.1, 0.2, 0.1], [0.1, 1.2, 0.3]],
        cell=[4.0, 4.0, 4.0],
        pbc=periodic,
    )
    a.info.update(N_alpha=1.7, N_beta=1.3, elec_temp=0.1, total_charge=0.0)
    return a


def small_model(**kwargs):
    return MLDFTB(
        [1],
        [1.0],
        hidden_irreps="2x0e+2x1o",
        num_interactions=1,
        matrix_feature_multiplicity=2,
        elec_temp_units="eV",
        **kwargs
    )


def args_from_example(tmp_path):
    a = extended_arg_parser().parse_args(
        [
            "--config=" + str(ROOT / "examples/mldftb/config.yaml"),
            "--train_file=" + str(tmp_path / "train.xyz"),
            "--hidden_irreps=2x0e+2x1o",
            "--num_interactions=1",
            "--work_dir=" + str(tmp_path),
        ]
    )
    check_config_conflicts(a)
    return a


def test_factory_optimizer_wrapper(tmp_path):
    args = args_from_example(tmp_path)
    model = build_model(args, AtomicNumberTable([1, 8]), np.zeros(2), None, None)
    options = get_param_options(model, args)
    optimizer = torch.optim.Adam(**options)
    assigned = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == {id(p) for p in model.parameters()}
    wrapper = make_model_wrapper(
        model, optimizer, dict(forces=True, virials=False, stress=False)
    )
    assert wrapper is not None
    from tests.test_mldftb import graph

    data = graph()
    data["node_attrs"] = torch.tensor([[1.0, 0.0]] * 3)
    before = [p.detach().clone() for p in model.hamiltonian.parameters()]
    # Simulate the evaluator, which switches off all parameter gradients.
    for p in model.parameters():
        p.requires_grad_(False)
    result = wrapper(model, data, training=True)
    loss = (result["energy"] - 1).square().sum() + result["forces"].square().sum()
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    optimizer.step()
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.hamiltonian.parameters())
    )
    with pytest.raises(ValueError, match="Missing effective nuclear"):
        build_model(args, AtomicNumberTable([6]), np.zeros(1), None, None)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("kind", ["internal", "free"])
def test_calculator_save_load_and_cache(tmp_path, periodic, kind):
    model = small_model(energy_kind=kind)
    path = tmp_path / "model.model"
    torch.save(model, path)
    a = atoms(periodic)
    a.calc = MLDFTBCalculator(path)
    energy = a.get_potential_energy(force_consistent=True)
    forces = a.get_forces()
    stress = a.get_stress()
    assert (
        np.isfinite(energy) and np.isfinite(forces).all() and np.isfinite(stress).all()
    )
    assert a.calc.check_state(a) == []
    np.testing.assert_array_equal(a.cell.array, np.eye(3) * 4)
    result = a.calc.results
    expected = result["energy"] + (result["entropy_energy"] if kind == "free" else 0)
    assert energy == pytest.approx(expected)
    assert a.get_charges().sum() == pytest.approx(0.0, abs=1e-10)
    # Changing only metadata must not reuse the geometry-only ASE cache.
    a.info["elec_temp"] = 0.3
    assert "elec_temp" in a.calc.check_state(a)
    assert np.isfinite(a.get_potential_energy(force_consistent=True))
    assert a.calc.results is not result
    a.info.update(N_alpha=1.0, N_beta=1.0, total_charge=1.0)
    assert a.get_charges().sum() == pytest.approx(1.0, abs=1e-10)
    a.info["external_field"] = np.array([0.1, 0.0, 0.0])
    before = a.get_potential_energy(force_consistent=True)
    a.info["external_field"][0] = 0.2
    after = a.get_potential_energy(force_consistent=True)
    assert before != pytest.approx(after, abs=1e-12)
    # Validate the saved model's ASE force against an energy finite difference.
    h = 1e-5
    force = a.get_forces()[1, 0]
    a.positions[1, 0] += h
    ep = a.get_potential_energy(force_consistent=True)
    a.positions[1, 0] -= 2 * h
    em = a.get_potential_energy(force_consistent=True)
    assert force == pytest.approx(-(ep - em) / (2 * h), abs=1e-6)


def test_calculator_per_atom_nuclei():
    a = atoms()
    a.arrays["zeff"] = np.ones(3)
    a.calc = MLDFTBCalculator(
        model=small_model(nuclear_charge_mode="from_data"),
        effective_nuclear_charges_key="zeff",
    )
    assert a.get_charges().sum() == pytest.approx(0.0, abs=1e-10)
    a.arrays["zeff"][0] = 2.0
    a.info["total_charge"] = 1.0
    assert "zeff" in a.calc.check_state(a)
    assert a.get_charges().sum() == pytest.approx(1.0, abs=1e-10)
    del a.info["N_alpha"]
    with pytest.raises(ValueError, match="Missing atoms.info"):
        a.get_potential_energy()


def test_training_command_and_reload(tmp_path):
    # Disposable fixtures only: no example training data is shipped.
    frames = []
    for i in range(6):
        a = atoms(periodic=bool(i % 2))
        a.positions[1, 0] += 0.05 * i
        a.info["REF_energy"] = 0.02 * i
        a.arrays["REF_forces"] = np.full((3, 3), 0.01 * i)
        if i % 2:
            a.arrays["REF_atomic_multipoles"] = np.zeros((3, 4))
        frames.append(a)
    write(tmp_path / "train.xyz", frames)
    config = yaml.safe_load((ROOT / "examples/mldftb/config.yaml").read_text())
    config.update(
        hidden_irreps="2x0e+2x1o",
        num_interactions=1,
        valid_fraction=0.34,
        batch_size=2,
        valid_batch_size=2,
        train_file=str(tmp_path / "train.xyz"),
        work_dir=str(tmp_path),
        E0s="{1: 0.0}",
        name="smoke",
    )
    config["mldftb_config"]["matrix_feature_multiplicity"] = 2
    config["heads"]["Default"]["arrays_keys"][
        "atomic_multipoles"
    ] = "REF_atomic_multipoles"
    config["train_schedule"][0].update(end=1)
    config["train_schedule"][0]["loss"]["atomic_multipoles"] = 1.0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    env = {**os.environ, "PYTHONPATH": str(ROOT), "MPLCONFIGDIR": str(tmp_path / "mpl")}
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_train.py"), "--config=" + str(path)],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
    assert run.returncode == 0, run.stdout[-18000:]
    saved = list((tmp_path / "checkpoints").glob("*.model"))
    assert saved
    a = atoms()
    a.calc = MLDFTBCalculator(saved[0])
    assert np.isfinite(a.get_potential_energy())
    assert np.isfinite(a.get_forces()).all()
    assert "hamiltonian" in run.stdout
    config["train_schedule"][0]["end"] = 2
    config["restart_latest"] = True
    path.write_text(yaml.safe_dump(config))
    resumed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_train.py"), "--config=" + str(path)],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
    assert resumed.returncode == 0, resumed.stdout[-12000:]
    assert (
        "Loading checkpoint" in resumed.stdout or "Loaded checkpoint" in resumed.stdout
    )
