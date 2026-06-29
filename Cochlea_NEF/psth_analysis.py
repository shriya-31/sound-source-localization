"""
Post-Stimulus Time Histogram (PSTH) for the cochlear NEF model.

A PSTH characterizes a neuron's (or population's) temporal response
pattern to a repeated stimulus: present the same tone burst many times,
align spike times to stimulus onset on each trial, bin them finely
(e.g. 1 ms bins), and average across trials. This is the standard tool
in auditory physiology for seeing onset transients, sustained/adapting
response, and offset responses -- features a single trial's raw spike
train is too noisy to show clearly.

Each trial here is a genuinely independent rebuild of the Nengo network
(fresh random neuron tuning + intrinsic spiking noise), exactly mirroring
how repeated presentations to a real, noisy biological neuron differ
trial-to-trial.
"""

import numpy as np
import matplotlib.pyplot as plt
import nengo

from cochlear_nef_model import CochlearModel


def make_tone_burst(fs, freq, total_dur=0.10, onset=0.02, burst_dur=0.05, amplitude=0.5):
    """
    A single trial's stimulus: silence, then a tone burst, then silence.
    onset: time (s) at which the tone burst starts -- this is t=0 for the PSTH.
    burst_dur: duration (s) of the tone.
    total_dur: total trial length (s); must be > onset + burst_dur to capture offset response.

    Note: `onset` should leave enough silent lead-in (recommend >= 0.03s)
    for the LIF ensemble's membrane potentials to reach their steady-state
    spontaneous firing rate before the stimulus arrives -- neurons start
    each trial at rest (V=0) and take a few membrane time constants to
    "warm up" to baseline. Without this lead-in, the pre-stimulus baseline
    in the PSTH shows a startup ramp that looks stimulus-related but is
    really just simulation initialization, not a real neural response.
    """
    t = np.arange(0, total_dur, 1 / fs)
    audio = np.zeros_like(t)
    mask = (t >= onset) & (t < onset + burst_dur)
    seg_t = t[mask] - onset
    tone = amplitude * np.sin(2 * np.pi * freq * seg_t)
    # short cosine ramp to avoid onset/offset clicks dominating the spectrum
    n_ramp = max(1, int(0.001 * fs))
    ramp = np.ones_like(tone)
    if len(tone) > 2 * n_ramp:
        ramp[:n_ramp] = 0.5 * (1 - np.cos(np.pi * np.arange(n_ramp) / n_ramp))
        ramp[-n_ramp:] = ramp[:n_ramp][::-1]
    audio[mask] = tone * ramp
    return t, audio


def run_trials(fs, channel_freq, stim_freq, n_trials, n_channels_total=16,
                low_freq=100, high_freq=8000, neurons_per_channel=40,
                onset=0.02, burst_dur=0.05, total_dur=0.10):
    """
    Run n_trials independent presentations of a tone burst at stim_freq,
    recording spikes from the channel whose CF is closest to channel_freq.
    Returns (spike_times_per_trial, sim_t, actual_cf).
    """
    t, audio = make_tone_burst(fs, stim_freq, total_dur, onset, burst_dur)

    # build once to find which channel index is closest to channel_freq
    probe_model = CochlearModel(fs=fs, n_channels=n_channels_total, low_freq=low_freq,
                                 high_freq=high_freq, neurons_per_channel=neurons_per_channel)
    ch_idx = int(np.argmin(np.abs(probe_model.cfs - channel_freq)))
    actual_cf = probe_model.cfs[ch_idx]

    all_spike_times = []
    sim_t_ref = None
    for trial in range(n_trials):
        model = CochlearModel(fs=fs, n_channels=n_channels_total, low_freq=low_freq,
                               high_freq=high_freq, neurons_per_channel=neurons_per_channel)
        model.set_audio(audio)
        net, probes = model.build()
        with nengo.Simulator(net, progress_bar=False) as sim:
            sim.run(total_dur)
        if sim_t_ref is None:
            sim_t_ref = sim.trange()
        spikes = sim.data[probes["spikes"][ch_idx]]  # (T, n_neurons), rate-coded (1/dt on spike, else 0)
        spike_mask = spikes > 0
        # collect (time, neuron) for every spike this trial, across all neurons in the ensemble
        rows, cols = np.where(spike_mask)
        all_spike_times.append(sim_t_ref[rows])

    return all_spike_times, sim_t_ref, actual_cf


def compute_psth(spike_times_per_trial, total_dur, bin_width=0.001, n_neurons=1):
    """
    Bin spike times into fixed-width bins and average across trials.
    Returns (bin_centers, rate_hz) where rate_hz is the mean firing rate
    per neuron per bin, averaged over trials -- the standard PSTH unit.
    """
    bins = np.arange(0, total_dur + bin_width, bin_width)
    n_trials = len(spike_times_per_trial)
    counts = np.zeros(len(bins) - 1)
    for trial_spikes in spike_times_per_trial:
        c, _ = np.histogram(trial_spikes, bins=bins)
        counts += c
    # rate (Hz) = (total spikes in bin) / (n_trials * n_neurons * bin_width)
    rate_hz = counts / (n_trials * n_neurons * bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, rate_hz


def main():
    fs = 20000
    n_trials = 30
    n_channels_total = 16
    neurons_per_channel = 40
    stim_freq = 15000       # tone burst frequency
    onset, burst_dur, total_dur = 0.04, 0.05, 0.15  # extra trailing silence avoids edge-of-trial bin artifact
    display_end = 0.13     # don't plot all the way to total_dur -- avoids the final-bin edge effect

    print(f"Running {n_trials} trials each for an on-CF channel and an off-CF channel "
          f"(this rebuilds the network independently each trial)...")

    # On-CF channel: tuned near the stimulus frequency -> strong response expected
    on_spikes, sim_t, on_cf = run_trials(
        fs, channel_freq=stim_freq, stim_freq=stim_freq, n_trials=n_trials,
        n_channels_total=n_channels_total, neurons_per_channel=neurons_per_channel,
        onset=onset, burst_dur=burst_dur, total_dur=total_dur)

    # Off-CF channel: tuned far from the stimulus (one octave+ away) -> weak/no response expected
    off_target_freq = stim_freq * 3.0
    off_spikes, _, off_cf = run_trials(
        fs, channel_freq=off_target_freq, stim_freq=stim_freq, n_trials=n_trials,
        n_channels_total=n_channels_total, neurons_per_channel=neurons_per_channel,
        onset=onset, burst_dur=burst_dur, total_dur=total_dur)

    bin_width = 0.001  # 1 ms bins, standard for auditory PSTHs
    t_on, rate_on = compute_psth(on_spikes, total_dur, bin_width, n_neurons=neurons_per_channel)
    t_off, rate_off = compute_psth(off_spikes, total_dur, bin_width, n_neurons=neurons_per_channel)

    # shift time axis so 0 = stimulus onset (standard PSTH convention)
    t_on_aligned = t_on - onset
    t_off_aligned = t_off - onset

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for ax, t_aligned, rate, cf, color in [
        (axes[0], t_on_aligned, rate_on, on_cf, "#1D9E75"),
        (axes[1], t_off_aligned, rate_off, off_cf, "#888780"),
    ]:
        ax.bar(t_aligned, rate, width=bin_width, color=color, align="center")
        ax.axvspan(0, burst_dur, color="black", alpha=0.06, zorder=0)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(burst_dur, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Rate (spikes/s)")
        ax.set_title(f"Channel CF = {cf:.0f} Hz  (stimulus = {stim_freq:.0f} Hz tone burst)")
        ax.set_xlim(-onset + 0.01, display_end - onset)

    axes[1].set_xlabel("Time relative to stimulus onset (s)")
    fig.suptitle(f"PSTH: {n_trials} trials, {bin_width*1000:.0f} ms bins, "
                 f"{neurons_per_channel} neurons/channel\n"
                 f"shaded region = tone burst (onset \u2192 offset)", fontsize=11)
    plt.tight_layout()

    out_path = "/mnt/user-data/outputs/cochlear_psth.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved PSTH figure to {out_path}")

    onset_window = (t_on_aligned >= 0) & (t_on_aligned < 0.01)
    sustained_window = (t_on_aligned >= 0.01) & (t_on_aligned < burst_dur)
    print(f"\nOn-CF channel ({on_cf:.0f} Hz): onset rate (0-10ms) = "
          f"{rate_on[onset_window].mean():.1f} Hz, sustained rate (10-{burst_dur*1000:.0f}ms) = "
          f"{rate_on[sustained_window].mean():.1f} Hz")
    print(f"Off-CF channel ({off_cf:.0f} Hz): mean rate during burst = "
          f"{rate_off[(t_off_aligned >= 0) & (t_off_aligned < burst_dur)].mean():.1f} Hz")


if __name__ == "__main__":
    main()
