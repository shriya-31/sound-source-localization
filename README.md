# sound-source-localization

A spiking neural network model of how a mouse works out **which direction a sound came from**,
built only from mechanisms the real auditory system is known to use.

## The idea in one paragraph

If a sound is off to your left, it arrives **louder in your left ear**, because your head blocks
some of the sound on its way to the right ear. That loudness gap is called the **Interaural Level
Difference (ILD)**, and it's the main cue mice use to localize sound. The brain measures it with a
simple circuit: a nucleus called the **LSO** receives *excitation* from the ear on its own side and
*inhibition* from the opposite ear, so its firing rate is effectively a subtraction between the two
ears. This repository simulates that circuit — as actual spiking neurons, in real time — from the
raw waveform through to a level-difference signal, and checks whether the simulated neurons behave
like real recorded ones.

The model is built with the **Neural Engineering Framework (NEF)** in
[Nengo](https://www.nengo.ai/), so every stage after the ear is a population of simulated
**leaky integrate-and-fire (LIF)** spiking neurons.

## Pipeline

```
mono sound + azimuth
      ↓
HRIR convolution            lso/spatializer.py      → separate left/right ear signals
      ↓
gammatone filterbank        Cochlea_NEF/            → split into 48 frequency channels (5–80 kHz)
      ↓
hair cell transduction      Cochlea_NEF/            → rectify + compress
      ↓
auditory nerve (spiking)    Cochlea_NEF/            → one LIF population per channel
      ↓
brainstem circuit           lso/brainstem.py        → SBC → MNTB → LSO, both sides
      ↓
ILD readout                 lso/results.py          → 48-channel level difference
```

Each stage in plain terms:

| Stage | What it is |
|---|---|
| **HRIR convolution** | An HRIR (Head-Related Impulse Response) is a recording of how a specific head and ear pair reshapes sound arriving from one direction. Convolving a mono sound with the left and right HRIRs for an angle produces realistic two-eared audio — this is what puts the loudness difference into the signal. |
| **Gammatone filterbank** | The inner ear is a frequency analyzer: different positions along it respond to different pitches. This is modelled as 48 bandpass filters whose centre frequencies follow the **Greenwood function**, the measured position-to-frequency map of a real cochlea. Mice hear far above humans, hence the 5–80 kHz range and the 500 kHz sample rate. |
| **Hair cell transduction** | Hair cells convert vibration into an electrical signal. Two things matter: they only respond to deflection in one direction (half-wave rectification), and they compress a huge range of loudness into a small range of response (power-law compression). |
| **Auditory nerve** | The first spiking stage — one population of LIF neurons per frequency channel. |
| **Brainstem circuit** | **SBC** (spherical bushy cells) relay each ear's signal. **MNTB** flips a signal from excitation into inhibition, which is how the *other* ear's input arrives as a subtraction. **LSO** combines same-side excitation with opposite-side inhibition, making each neuron a level-difference detector. |
| **ILD readout** | The difference between right and left LSO activity, per frequency channel. Available both as an NEF-decoded value and as true per-neuron firing rates in Hz. |

`_L` and `_R` throughout the code mean left and right.

## Layout

```
Cochlea_NEF/     ear model: filterbank, hair cells, auditory nerve, PSTH validation, demo plots
lso/             brainstem model: config, circuit, stimuli, spatialization, experiments, plots
validation/      comparison of model neurons against published electrophysiology
data/            bundled mouse HRIR dataset
results/         saved figures and raw sweep outputs
mso.py           standalone prototype of the timing-based (ITD/MSO) pathway — not wired in
```

Key files:

- **`lso/config.py`** — every parameter in one dataclass. Read this first.
- **`lso/brainstem.py`** — the circuit itself; `BrainstemModel.run(left, right)` is the main entry point.
- **`lso/experiment.py`** — sweeps a range of azimuths and collects ILD at each.
- **`lso/decoder.py`** — maps the 48-channel ILD vector to a single estimated angle, using an NEF
  ensemble whose output weights are solved by regularized least squares. *Complete, but not
  currently called from anywhere in the pipeline.*
- **`validation/`** — fits sigmoids to each model neuron's rate-vs-ILD curve to extract **dynamic
  range** and **ILD50** (the level difference at which the neuron hits half its maximum rate), then
  compares those against published values from real neurons.

## Setup

Python 3.11. Core dependencies are numpy, scipy, matplotlib and `nengo==4.1.0`.

```bash
pip install -r requirements.txt
```

Verify the install with the smoke test, which runs the full spiking pipeline on an identical tone
in both ears and needs no external data:

```bash
python lso/env_check.py
```

**Note on data:** the mouse HRIR dataset ships with the repo at
`data/hrir_fixed_dataset.npz`, so spatialized audio runs out of the box — load it with
`load_hrir()` from `lso/hrir.py`. Be aware it contains no genuine left-ear measurement: the
left ear is an exact mirror of the right. The original recordings are on
[figshare article 11691615](https://figshare.com/articles/dataset/Mouse_Head_Related_Transfer_Function_processed_data/11691615)
(~870 MB per mouse); `load_hrir()` takes a path to that `.mat` too, and it's the better choice
for anything depending on real left/right asymmetry.

The older `ild-be.ipynb` still points at an untracked external `../hrtf/*.mat` and has its ear
indices swapped — treat it as historical, not a working entry point.

**Note on imports:** modules use bare, working-directory-relative imports (e.g.
`from config import SimulationConfig`), so `lso/` and `Cochlea_NEF/` both need to be on
`sys.path`. See the top of `lso/env_check.py` for the pattern.

## Further reading

- `OVERVIEW.md` — the project's goals, how the work maps onto machine learning concepts, and the
  current state of each component in more depth.
- `PROGRESS.md` — running log of what has been done in each phase of work.
