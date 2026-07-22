"""
run_within_channel_population.py

Alternative, much cheaper population axis to run_population.py's
48-channel loop: treat the ~200 individual neurons *within one
channel* as the population, instead of looping across channels.

Design choice (see Step 7 of the plan): all neurons share ONE ipsi --
the channel-mean threshold+20 that find_centre_and_bandwidth already
computes for the channel-level fit. That means the 21-point contra
sweep only has to run once per window; each neuron's own rate is read
out of that already-cached data (LSOWrapper._cache) and fit
independently, at zero additional simulation cost. The tradeoff: this
isn't a fully independent per-neuron measurement the way a real
recorded neuron (referenced only to its own threshold) is -- the
shared ipsi is itself derived from averaging over all 200 neurons,
including whichever one is currently being fit. Disclosed
simplification, not a hidden one.

This is a *different*, complementary question from run_population.py:
within-CF (same channel) neuron-to-neuron variability, not cross-CF
(across channels) variability like Fisch's real n=13 sample likely has.
"""

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "lso"))

import numpy as np
from scipy.optimize import curve_fit

from level_difference_wrapper import LSOWrapper, WindowedView
from level_difference_coding import find_centre_and_bandwidth, logsig
from plot_population import plot_c_and_d

FIELDNAMES = ["neuron", "dynamic_range_tst", "dynamic_range_onset", "ild50_tst", "ild50_onset"]


def _per_neuron_rate_matrix(wrapper, ipsi, freq, window, channel, contras):
    """
    Sweep contras at the fixed, shared ipsi (hits LSOWrapper's cache --
    these are the exact simulations find_centre_and_bandwidth already
    ran to fit the channel-level curve) and return a (len(contras),
    n_neurons) matrix of that channel's per-neuron rates at each step.
    """
    rows = []
    for contra in contras:
        wrapper.get_spike_rate(ipsi, contra, freq, window=window)  # ensures cached
        results = wrapper._cache[(ipsi, contra, freq)]
        neuron_rates = (results.lso_L_neuron_rates_tst if window == "tst"
                         else results.lso_L_neuron_rates_onset)[channel]
        rows.append(neuron_rates)
    return np.array(rows)


def _fit_neuron_curves(contras, rate_matrix, ipsi):
    """
    rate_matrix: (len(contras), n_neurons). Fits logsig per neuron
    column (same curve_fit pattern as find_centre_and_bandwidth).
    Returns a list of {"dynamic_range", "ild50"} dicts, or None for
    neurons whose fit fails.
    """
    fits = []
    for neuron_idx in range(rate_matrix.shape[1]):
        rates = rate_matrix[:, neuron_idx]
        try:
            popt, _ = curve_fit(logsig, contras, rates, p0=[-1, 0, float(np.max(rates))],
                                 bounds=([-10, -100, 0], [0, 100, 1000]))
        except (RuntimeError, ValueError) as e:
            print(f"[within-channel] neuron {neuron_idx} failed to fit: {e}")
            fits.append(None)
            continue
        slope, bias, _ = popt
        centre = -bias / slope
        bandwidth = -2 * np.log(9) / slope
        fits.append({"dynamic_range": bandwidth, "ild50": ipsi - centre})
    return fits


def run_within_channel_population(channel, out_path=None):
    """
    For the given channel, fit each of its individual neurons'
    (ILD50, dynamic range) for both TST and onset windows, using one
    shared ipsi per window (the channel-level threshold+20). Writes
    validation/population_results_ch{channel}.csv (or out_path).
    """
    if out_path is None:
        out_path = Path(__file__).resolve().parent / f"population_results_ch{channel}.csv"

    wrapper = LSOWrapper()
    freq = float(wrapper.cfs[channel])
    contras = np.linspace(0, 100, 21)  # matches find_centre_and_bandwidth's own sweep

    rows_by_neuron = {}
    for window in ("tst", "onset"):
        view = WindowedView(wrapper, window)
        # Channel-level fit first: gives us the shared ipsi, and as a
        # side effect populates the cache for the whole contra sweep.
        _, _, ipsi = find_centre_and_bandwidth(view, frequency=freq, plot=False)

        rate_matrix = _per_neuron_rate_matrix(wrapper, ipsi, freq, window, channel, contras)
        fits = _fit_neuron_curves(contras, rate_matrix, ipsi)

        for neuron_idx, fit in enumerate(fits):
            if fit is None:
                continue
            row = rows_by_neuron.setdefault(neuron_idx, {"neuron": neuron_idx})
            row[f"dynamic_range_{window}"] = fit["dynamic_range"]
            row[f"ild50_{window}"] = fit["ild50"]

    # Only keep neurons where both windows fit successfully.
    complete_rows = [r for r in rows_by_neuron.values()
                      if "dynamic_range_tst" in r and "dynamic_range_onset" in r]
    skipped = len(rows_by_neuron) - len(complete_rows)
    print(f"[within-channel] channel {channel}: {len(complete_rows)} neurons fit "
          f"successfully in both windows, {skipped} skipped")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(complete_rows)
    print(f"[within-channel] wrote {out_path}")

    return complete_rows


if __name__ == "__main__":
    CHANNEL = 24  # len(wrapper.cfs) // 2 (48 channels total) -- matches level_difference_coding.py's Case 3 default
    rows = run_within_channel_population(CHANNEL)
    out_dir = Path(__file__).resolve().parent
    plot_c_and_d(
        rows,
        out_path=out_dir / f"fig_c_d_ch{CHANNEL}_neurons.png",
        population_label=f"neurons in channel {CHANNEL}",
    )
