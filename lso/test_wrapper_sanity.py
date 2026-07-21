"""
test_wrapper_sanity.py

Sanity checks for LSOWrapper before trusting it in the Fig. 3.25
analysis pipeline. Confirms:
  1. Ipsi-dominant input produces a higher LSO rate than contra-dominant
     input (excitation from ipsi SBC, inhibition via contra-driven MNTB).
  2. Increasing contra level at fixed ipsi level monotonically suppresses
     the rate -- the basic ILD tuning curve shape Fig. 3.25 relies on.
  3. The channel nearest a test frequency actually has the (near-)highest
     response to a tone at that frequency, i.e. the tonotopic channel
     selection in get_spike_rate is picking the right channel. Reports
     both the old NEF-decoded profile and the new real spike-rate
     profile side by side, to check whether the channel-gain caveat
     found under the decoded value still shows up under real rates.
  4. Individual neurons within one channel's population actually differ
     from each other (not identical rates), and their spread is visible
     alongside the population mean -- addresses Bryan's feedback to
     "try a few different neurons of the same channel."
"""

import numpy as np
from level_difference_wrapper import LSOWrapper

wrapper = LSOWrapper()
TEST_CH = 39  # CF ~50078 Hz -- high-frequency channel, contrast to ch 24
freq = float(wrapper.cfs[TEST_CH])
print(f"Test frequency: {freq:.0f} Hz (channel {TEST_CH})")

# --- Test 1: excitation direction ---
rate_ipsi_dominant = wrapper.get_spike_rate(ipsi_level=70, contra_level=0, frequency=freq)
rate_contra_dominant = wrapper.get_spike_rate(ipsi_level=0, contra_level=70, frequency=freq)
print(f"\n[1] ipsi=70/contra=0  -> rate={rate_ipsi_dominant:.4f}")
print(f"[1] ipsi=0/contra=70  -> rate={rate_contra_dominant:.4f}")
print("[1] PASS" if rate_ipsi_dominant > rate_contra_dominant else "[1] FAIL")

# --- Test 2: monotonic suppression as contra level increases ---
print("\n[2] Fixed ipsi=70, sweeping contra:")
contra_levels = [0, 45, 90]
rates = []
for contra in contra_levels:
    r = wrapper.get_spike_rate(ipsi_level=70, contra_level=contra, frequency=freq)
    rates.append(r)
    print(f"    contra={contra:>3} dB -> rate={r:.4f}")
monotonic = all(rates[i] >= rates[i + 1] for i in range(len(rates) - 1))
print("[2] PASS (monotonic non-increasing)" if monotonic else "[2] FAIL (not monotonic)")

# --- Test 3: tonotopic channel selection ---
ipsi_tone = wrapper._level_to_amplitude(70) * wrapper.stimulus_gen.generate_pure_tone(freq)
contra_tone = wrapper._level_to_amplitude(0) * wrapper.stimulus_gen.generate_pure_tone(freq)
results = wrapper.brainstem.run(left_signal=ipsi_tone, right_signal=contra_tone)
expected_ch = int(np.argmin(np.abs(wrapper.cfs - freq)))
actual_ch = int(np.argmax(results.lso_L))
print(f"\n[3] Expected peak channel: {expected_ch} (CF={wrapper.cfs[expected_ch]:.0f} Hz)")
print(f"[3] Actual peak channel (decoded lso_L):   {actual_ch} (CF={wrapper.cfs[actual_ch]:.0f} Hz)")
actual_ch_hz = int(np.argmax(results.lso_L_rate_hz))
print(f"[3] Actual peak channel (real lso_L_rate_hz): {actual_ch_hz} (CF={wrapper.cfs[actual_ch_hz]:.0f} Hz)")
print("[3] PASS" if abs(actual_ch_hz - expected_ch) <= 2 else "[3] FAIL")

print("\n[3] Full per-channel profile (channel, CF, decoded lso_L, real lso_L_rate_hz), sorted by real rate:")
order = np.argsort(results.lso_L_rate_hz)[::-1]
for ch in order:
    marker = "  <-- expected" if ch == expected_ch else ""
    print(f"    ch {ch:2d}  CF={wrapper.cfs[ch]:8.0f} Hz  decoded={results.lso_L[ch]:.4f}  rate_hz={results.lso_L_rate_hz[ch]:7.2f}{marker}")

# --- Test 4: individual neuron variability within one channel ---
test_ch = TEST_CH
neuron_rates = results.lso_L_neuron_rates[test_ch]
print(f"\n[4] Channel {test_ch} (CF={wrapper.cfs[test_ch]:.0f} Hz), first 10 of {len(neuron_rates)} neurons:")
for i, r in enumerate(neuron_rates[:10]):
    print(f"    neuron {i:2d}: {r:7.2f} Hz")
print(f"    population mean: {neuron_rates.mean():7.2f} Hz  (matches lso_L_rate_hz[{test_ch}]={results.lso_L_rate_hz[test_ch]:.2f})")
print(f"    population std:  {neuron_rates.std():7.2f} Hz")
print("[4] PASS (neurons differ)" if neuron_rates.std() > 0 else "[4] FAIL (all neurons identical)")
