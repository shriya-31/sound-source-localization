"""
hrir.py
Loading of the mouse HRIR dataset used for binaural spatialization.

Two sources are supported:

- ``data/hrir_fixed_dataset.npz`` (bundled in this repo) -- derived from the
  figshare ``.mat`` below, keeping only the right-ear measurements and
  mirroring them to produce the left ear. Self-contained, so it is the
  default, but note the left ear is synthetic and the dataset is
  therefore perfectly left/right symmetric.
- The original ``.mat`` from figshare (Ito, Si, Feldheim & Litke, mouse
  HRTF, article 11691615), e.g.
  ``2018-08-15_1000samp_ILDITDmod_correct_noitd_ex_ILD.mat``. ~870 MB and
  not tracked here. Contains genuine left-ear measurements, so prefer it
  for anything that depends on real left/right asymmetry.

Both return the same ``(2, elevation, azimuth, samples)`` layout that
``Spatializer`` expects, with ear index 0 = left and 1 = right, elevation
index 0 = the 0-degree horizontal plane, and 200 azimuths at 1.8-degree
steps. That layout is documented by the dataset authors in
``Demo_PlotHRTFs.m`` of shixnya/SoundLocalizationResearch.
"""

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NPZ = REPO_ROOT / "data" / "hrir_fixed_dataset.npz"

# Variable the project reads out of the figshare .mat. Note the dataset
# page documents allimps, _noitd, _nospec, _ILDequal, _percep, _present
# and _ILDcorrected, but not _reflect -- carried over from main.ipynb.
MAT_VARIABLE = "allimps_reflect"


def load_hrir(path=None, mat_variable=MAT_VARIABLE):
    """
    Load an HRIR array.

    Parameters
    path: str | Path | None
        ``.npz`` or ``.mat`` file. Defaults to the bundled ``.npz``.
    mat_variable: str
        Variable to read when loading a ``.mat``; ignored for ``.npz``.

    Returns
    np.ndarray
        HRIR array of shape (2, elevation, azimuth, samples).
    """
    path = Path(path) if path is not None else DEFAULT_NPZ
    if not path.exists():
        raise FileNotFoundError(
            f"HRIR dataset not found at {path}. The bundled dataset is "
            f"data/{DEFAULT_NPZ.name}; the original .mat is on figshare "
            "(article 11691615)."
        )

    if path.suffix == ".npz":
        hrir = np.load(path)["hrir"]
    elif path.suffix == ".mat":
        import scipy.io as sio

        hrir = sio.loadmat(path)[mat_variable]
    else:
        raise ValueError(f"Unsupported HRIR file type: {path.suffix}")

    if hrir.ndim != 4 or hrir.shape[0] != 2:
        raise ValueError(
            f"Expected HRIR shape (2, elevation, azimuth, samples), got {hrir.shape}"
        )
    return hrir
