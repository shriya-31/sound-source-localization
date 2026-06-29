"""
Demo: drive the cochlear NEF model with a sweep / chord and visualize
the tonotopic response -- a spike raster (sorted by channel CF, low to
high) and the resulting excitation pattern (firing rate vs CF), which is
the standard way auditory physiologists display cochlear responses.
"""

import numpy as np
import matplotlib.pyplot as plt
import nengo

from cochlear_nef_model import CochlearModel


def make_stimulus(fs, duration=0.5):
    """A logarithmic frequency sweep (chirp) from low to high, followed by
    a three-tone chord. The sweep is the cleanest way to see the cochlea's
    traveling wave: the locus of peak excitation should move smoothly from
    low channels to high channels as frequency rises, since every part of
    the spectrum gets swept through continuously (no gaps between discrete
    test tones for a channel to silently fall between)."""
    t_sweep = np.arange(0, duration * 0.7, 1 / fs)
    f0, f1 = 100, 20000
    # logarithmic chirp: log-spaced frequency tracks the Greenwood map almost linearly
    k = (f1 / f0) ** (1 / t_sweep[-1])
    phase = 2 * np.pi * f0 * (k ** t_sweep - 1) / np.log(k)
    sweep = np.sin(phase)
    # onset/offset ramp to avoid a click
    ramp = np.ones_like(sweep)
    n_ramp = int(0.005 * fs)
    ramp[:n_ramp] = np.linspace(0, 1, n_ramp)
    ramp[-n_ramp:] = np.linspace(1, 0, n_ramp)
    sweep *= ramp

    t_chord = np.arange(0, duration * 0.3, 1 / fs)
    chord = sum(np.sin(2 * np.pi * f * t_chord) for f in [250, 5000, 20000])
    chord *= ramp[:len(t_chord)] if len(ramp) >= len(t_chord) else 1.0
    chord_ramp = np.ones_like(chord)
    chord_ramp[:n_ramp] = np.linspace(0, 1, n_ramp)
    chord_ramp[-n_ramp:] = np.linspace(1, 0, n_ramp)
    chord *= chord_ramp

    audio = np.concatenate([sweep, chord]) * 0.3
    t = np.arange(len(audio)) / fs
    return t, audio


def main():
    fs = 20000
    f0, f1 = 100, 20000
    n_channels = 32
    model = CochlearModel(fs=fs, n_channels=n_channels, low_freq=80, high_freq=80000,
                           neurons_per_channel=25)

    t, audio = make_stimulus(fs, duration=0.5)
    model.set_audio(audio)
    net, probes = model.build()

    print(f"Running NEF simulation: {n_channels} channels x {model.neurons_per_channel} neurons "
          f"= {n_channels * model.neurons_per_channel} total neurons, {len(t)} timesteps...")

    with nengo.Simulator(net, progress_bar=False) as sim:
        sim.run(t[-1])

    # --- Build spike raster data, sorted tonotopically (already in CF order) ---
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                              gridspec_kw={"height_ratios": [1, 2.2, 1.2]})

    # Panel 1: stimulus waveform
    axes[0].plot(t, audio, color="#444", linewidth=0.5)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Stimulus: logarithmic chirp {f0} to {f1} Hz, then 250+5000+20000 Hz chord")

    # Panel 2: spike raster, rows ordered by channel (bottom=low CF, top=high CF)
    sim_t = sim.trange()
    for ch in range(n_channels):
        spikes = sim.data[probes["spikes"][ch]]  # shape (T, n_neurons), rate-coded
        spike_mask = spikes > 0
        spike_rows, spike_cols = np.where(spike_mask)
        spike_times = sim_t[spike_rows]
        # each dot's row jitter spans the channel's band; one dot per (time, neuron) spike
        y = ch + (spike_cols / spikes.shape[1]) - 0.5
        axes[1].scatter(spike_times, y, s=2, color="black", alpha=0.5, marker="|")

    axes[1].set_ylabel("Channel (low CF \u2192 high CF)")
    axes[1].set_yticks(np.arange(0, n_channels, 4))
    axes[1].set_yticklabels([f"{model.cfs[i]:.0f} Hz" for i in range(0, n_channels, 4)])
    axes[1].set_ylim(-1, n_channels)
    axes[1].set_title("Auditory nerve spike raster (one ensemble per cochlear channel)")

    # Panel 3: time-varying population rate per channel as an image (cochleagram).
    # Use a sqrt color scale (global, not per-row normalized) -- this keeps
    # relative magnitude meaningful across channels while still making the
    # naturally-weaker high-CF responses visible, since raw linear scaling
    # is dominated by the loudest band.
    rate_image = np.zeros((n_channels, len(sim_t)))
    kernel = np.ones(80) / 80
    for ch in range(n_channels):
        spikes = sim.data[probes["spikes"][ch]].mean(axis=1)  # mean rate across neurons in ensemble
        rate_image[ch] = np.convolve(spikes, kernel, mode="same")

    im = axes[2].imshow(np.sqrt(rate_image), aspect="auto", origin="lower",
                         extent=[sim_t[0], sim_t[-1], 0, n_channels],
                         cmap="inferno")
    axes[2].set_ylabel("Channel")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Cochleagram (sqrt-scaled population firing rate per channel)")
    fig.colorbar(im, ax=axes[2], label="sqrt(Rate) [Hz\u00b9\u141f\u00b2]", fraction=0.025, pad=0.01)

    plt.tight_layout()
    out_path = "./cochlear_nef_response.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")

    # Print a quick summary excitation pattern for the chord segment (last 30% of audio)
    chord_start = t[-1] * 0.75
    chord_end = t[-1] * 0.95
    chord_mask = (sim_t >= chord_start) & (sim_t < chord_end)
    chord_rates = np.array([
        sim.data[probes["spikes"][ch]][chord_mask].mean() for ch in range(n_channels)
    ])
    top3 = np.argsort(chord_rates)[-3:][::-1]
    print("\nDuring chord (250+1500+5000 Hz), top 3 responding channels:")
    for ch in top3:
        print(f"  channel {ch}: CF={model.cfs[ch]:7.1f} Hz, rate={chord_rates[ch]:6.1f} Hz")


if __name__ == "__main__":
    main()
