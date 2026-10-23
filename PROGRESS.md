# Progress Log

Running record of what was done, when, and why. Newest phase at the top.
Background and the full gap analysis live in `OVERVIEW.md`.

---

## Phase 2 — Repo cleanup (2026-10-23)

Housekeeping so the repo is self-contained and a clean checkout runs.

- `hrir_fixed_dataset.npz` moved to `data/`.
- Sweep outputs moved to `results/`.
- Removed the stray 1-byte file `lso/-`.
- `.gitignore` now excludes `__pycache__/`.
- Committed work that had been sitting untracked: `OVERVIEW.md`, `README.md`,
  `lso/closed_loop.py` (Alisa's code — still not run or wired up).

---

## Phase 1 — End-to-end run to the 48-channel ILD vector (2026-10-23)

**Goal.** Run the pipeline from stimulus through to the per-channel neural ILD
(`LSO_R − LSO_L`), to confirm it works and see what the cue actually looks like.

### Bug found and fixed

`BrainstemModel.run()` raised `KeyError: 'lso_R_neurons'` on **every** call. Commit
`c5d8119` had added the `lso_R` rate and `ild_neural_*` computations without the matching
per-neuron probes. This broke the end-to-end pipeline *and* `LSOWrapper`, so the validation
scripts were also unrunnable. Fixed by mirroring the existing `lso_L` probe block
(commit `a3abe02`). The committed channel-24 validation results predate the breakage.

### Changes made

| Change | Why |
|---|---|
| New `lso/hrir.py` | One `load_hrir()` entry point for both the bundled `.npz` and the figshare `.mat`. Documents the array layout (ear 0 = left, elevation 0 = horizontal plane, 200 azimuths at 1.8°) so it doesn't have to be re-derived. |
| `lso/main.ipynb` cell 0 | Added `sys.path` setup. The notebook imported `cochlear_nef_model` from `Cochlea_NEF/` without putting it on the path, so it failed on a clean checkout. |
| `lso/main.ipynb` cell 1 | Now `load_hrir()` instead of a hardcoded path to an untracked external `.mat`. |

### Dataset provenance (resolved)

The `.mat` the notebooks referenced is from **figshare article 11691615** (Ito, Si, Feldheim
& Litke — HRTFs of 3 female CBA/CaJ mice). The three `.mat` files are three different
*animals*; this project uses `2018-08-15`. The variable read is `allimps_reflect`, which is
**not** documented on the dataset page — still worth asking Alisa about.

`data/hrir_fixed_dataset.npz` is derived from that file: right-ear data only, with the left
ear produced by mirroring. Verified directly — left is an *exact* azimuth-mirror of right
(cosine similarity 1.0000). So it carries no genuine left-ear measurement and is perfectly
left/right symmetric. Fine for exploration; prefer the real `.mat` for decoder training,
where genuine asymmetry may be what separates front from back.

The authors' own `Demo_PlotHRTFs.m` documents the array layout, which confirmed that
`lso/spatializer.py` (ear 0 = left) and `config.elevation_idx = 0` (horizontal plane) are
both correct. The older `ild-be.ipynb` has the ears swapped and uses elevation 5 — any
figure from it is mirrored.

### Result

Full sweep, 101 azimuths from −90° to +90°, broadband noise, seed 1.
**7.6 min total, 4.5 s/angle.** Outputs in `results/`.

- Broadband ILD rises monotonically, −0.15 at −90° to +0.13 at +90°
- **Correlation with azimuth: +0.989**; 95% monotonic
- Steep in the high-frequency channels (19.6 / 31 / 50 kHz), nearly flat at 5 kHz — the
  expected head-shadow physics, and not something the model was tuned to produce

### Open points from this run

1. **~17% left/right asymmetry.** Antisymmetry residual 0.175; ILD at 0° is −0.005, not 0.
   The HRIRs are exactly mirrored, so this comes from the model itself — the left and right
   ensembles draw different random neuron parameters. Systematic, not noise (matches an
   independent two-angle check). Seeding the network would remove it.
2. **Channel 39 (50 kHz) is erratic** — deep dip near −40°. Either genuine HRTF spectral
   notches or single-trial noise; a second seed would distinguish them.
3. **Front hemisphere only.** `ICCircularDecoder` spans 0–360°, so training needs a
   full-circle sweep — which is also where front/back ambiguity gets settled.

---

## Next

Full-circle sweep doubles as the decoder training set, so it's the natural next step:
sweep 0–360° → collect `ild_neural_tst` vectors → train `ICCircularDecoder` → check
front/back confusion → then `lso/closed_loop.py` runs nearly as-is, minus the `report()`
body still needed from Alisa.
