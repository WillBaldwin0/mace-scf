"""Spectral summaries and display functions; never perform model evaluation."""

import numpy as np


def spin_channels(spectrum):
    channels = spectrum["channels"]
    if spectrum["spin_mode"] == "RKS":
        return {"alpha": channels["shared"], "beta": channels["shared"]}
    return channels


def gap_summary(spectrum):
    occupied, empty = [], []
    for channel in spectrum["channels"].values():
        e, f = channel["eigenvalues"], channel["occupations"]
        occupied.extend(e[f > 0.5])
        empty.extend(e[f < 0.5])
    if not occupied or not empty:
        return dict(
            value=float("nan"),
            occupied_edge=float("nan"),
            empty_edge=float("nan"),
            available=False,
        )
    hi, lo = float(max(occupied)), float(min(empty))
    return dict(value=lo - hi, occupied_edge=hi, empty_edge=lo, available=True)


def display_dos(spectrum, width=0.2, points=2000):
    if not np.isfinite(width) or width <= 0 or points < 2:
        raise ValueError(
            "Positive finite display width and at least two grid points required"
        )
    all_e = np.concatenate([c["eigenvalues"] for c in spectrum["channels"].values()])
    grid = np.linspace(all_e.min() - 5 * width, all_e.max() + 5 * width, points)
    result = {"energy": grid}
    for name, c in spin_channels(spectrum).items():
        dos, filled = np.zeros(points), np.zeros(points)
        for start in range(0, len(c["eigenvalues"]), 128):
            e = c["eigenvalues"][start : start + 128]
            kernel = np.exp(-0.5 * ((grid[:, None] - e) / width) ** 2) / (
                width * np.sqrt(2 * np.pi)
            )
            dos += kernel.sum(1)
            filled += kernel @ c["occupations"][start : start + 128]
        result[name + "_dos"], result[name + "_occupied_dos"] = dos, filled
    return result


def plot_spectrum(path, spectrum, charge, width=0.2):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FuncFormatter, AutoMinorLocator

    curves = display_dos(spectrum, width)
    energy = curves["energy"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 7), sharey=True)
    for name, sign, color, marker in [
        ("alpha", -1, "tab:blue", "o"),
        ("beta", 1, "tab:orange", "x"),
    ]:
        c = spin_channels(spectrum)[name]
        e, f = c["eigenvalues"], c["occupations"]
        axes[0].plot(sign * curves[name + "_dos"], energy, color=color, label=name)
        axes[0].fill_betweenx(
            energy, 0, sign * curves[name + "_occupied_dos"], color=color, alpha=0.3
        )
        axes[1].scatter(f, e, s=12, marker=marker, color=color, label=name)
        mu = float(c["chemical_potential"])
        if np.isfinite(mu):
            for ax in axes:
                ax.axhline(
                    mu, color=color, linestyle="--" if name == "alpha" else ":", lw=0.8
                )
    for name, c in spectrum["channels"].items():
        unique, counts = np.unique(c["eigenvalues"], return_counts=True)
        # Exact count before/after each jump, including degeneracies.
        x, y = [0.0], [energy[0]]
        count = 0
        for e, n in zip(unique, counts):
            x.extend([count, count + n])
            y.extend([e, e])
            count += n
        x.append(count)
        y.append(energy[-1])
        axes[2].plot(x, y, label=name)
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{abs(x):g}"))
    axes[0].set(
        xlabel="DOS: alpha left | beta right (states/eV)",
        ylabel="Eigenvalue (eV)",
        ylim=(energy[0], energy[-1]),
    )
    axes[1].set(xlabel="Per-spin occupancy", xticks=[0, 0.5, 1], xlim=(-0.05, 1.05))
    axes[2].set(xlabel="Cumulative states (index + 1 after jump)")
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    axes[2].xaxis.set_minor_locator(AutoMinorLocator(2))
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[2].grid(True, which="minor", alpha=0.15)
    pops = [spin_channels(spectrum)[s]["occupations"].sum() for s in ("alpha", "beta")]
    fig.suptitle(
        f"Charge {charge:+g} e; Nα={pops[0]:g}, Nβ={pops[1]:g}; Gaussian width {width:g} eV"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return curves
