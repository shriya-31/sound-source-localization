"""
plot_population.py

Reads validation/population_results.csv (produced by run_population.py)
and plots a Fig. 3.25 C&D-style figure: two panels (dynamic range,
ILD50), each showing paired TST/Onset columns -- individual channels
as jittered dots, mean +/- SD as a black diamond, and Fisch (2025)'s
own reported mean +/- SD overlaid in red for direct comparison. This
is the cheap half of Step 6; re-run any time population_results.csv
changes, no simulation required.
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_CSV = Path(__file__).resolve().parent / "population_results.csv"
OUT_PNG = Path(__file__).resolve().parent / "fig_c_d_model.png"

# Fisch (2025) Fig. 3.25 C&D, n=13 real neurons -- see Fisch2025.pdf
FISCH = {
    "dynamic_range": {"tst": (40.4, 16.6), "onset": (17.5, 0.9)},
    "ild50": {"tst": (10.9, 15.2), "onset": (1.6, 0.9)},
}


def load_results(path=RESULTS_CSV):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: (int(v) if k in ("channel", "neuron") else float(v)) for k, v in row.items()})
    return rows


def _plot_metric(ax, rows, metric, ylabel, rng):
    tst = np.array([r[f"{metric}_tst"] for r in rows])
    onset = np.array([r[f"{metric}_onset"] for r in rows])

    x_tst = 0 + rng.normal(0, 0.05, size=len(tst))
    x_onset = 1 + rng.normal(0, 0.05, size=len(onset))
    ax.scatter(x_tst, tst, color="gray", alpha=0.6, zorder=2, label="Model (per channel)")
    ax.scatter(x_onset, onset, color="gray", alpha=0.6, zorder=2)

    ax.errorbar([0], [tst.mean()], yerr=[tst.std()], fmt="D", color="black",
                capsize=4, zorder=3, label="Model mean +/- SD")
    ax.errorbar([1], [onset.mean()], yerr=[onset.std()], fmt="D", color="black",
                capsize=4, zorder=3)

    fisch_tst_mean, fisch_tst_sd = FISCH[metric]["tst"]
    fisch_onset_mean, fisch_onset_sd = FISCH[metric]["onset"]
    ax.errorbar([0.2], [fisch_tst_mean], yerr=[fisch_tst_sd], fmt="D", color="red",
                capsize=4, zorder=3, label="Fisch (2025), n=13")
    ax.errorbar([1.2], [fisch_onset_mean], yerr=[fisch_onset_sd], fmt="D", color="red",
                capsize=4, zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["TST", "Onset"])
    ax.set_xlim(-0.4, 1.6)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def plot_c_and_d(rows, out_path=OUT_PNG, seed=0, population_label="model channels"):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    _plot_metric(axes[0], rows, "dynamic_range", "Dynamic range [dB]", rng)
    _plot_metric(axes[1], rows, "ild50", "ILD50 [dB]", rng)
    axes[0].set_title(f"C: Dynamic range (n={len(rows)} {population_label})")
    axes[1].set_title(f"D: ILD50 (n={len(rows)} {population_label})")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Model vs. Fisch (2025) Fig. 3.25 C&D")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    rows = load_results()
    print(f"Loaded {len(rows)} channels from {RESULTS_CSV}")
    plot_c_and_d(rows)
