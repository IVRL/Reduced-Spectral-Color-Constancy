"""
CIE illuminant SPD loaders for the 400–700 nm / 10 nm grid.

Provides loaders for daylight (D-series), fluorescent (F1–F12), illuminant A,
and LED illuminants, plus a unified load_illuminants() entry point.

The train/test illuminant name lists (ILL_TRAIN_SET, ILL_TEST_SET) define the
standard split used across all experiments.
"""

import csv
import json
import os
from collections.abc import Iterable

import numpy as np
import pandas as pd
import colour


_ILLUMINANT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_raw", "illuminant_spds")

ILL_TRAIN_SET = ['D50', 'D65', 'F1', 'F10', 'F12', 'F3', 'LED-B1', 'LED-B4', 'LED-RGB1', 'LED-V1']
ILL_TEST_SET  = ['D50', 'D55', 'D60', 'D65', 'D75', 'D93',
                 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12',
                 'A',
                 'LED-B1', 'LED-B2', 'LED-B3', 'LED-B4', 'LED-B5',
                 'LED-BH1', 'LED-RGB1', 'LED-V1', 'LED-V2']


# ── Utility ───────────────────────────────────────────────────────────────────

def spd_to_xy(spd: np.ndarray, wavelengths: np.ndarray) -> tuple[float, float]:
    """Convert an SPD sampled on `wavelengths` to CIE 1931 xy chromaticity."""
    shape = colour.SpectralShape(wavelengths[0], wavelengths[-1], wavelengths[1] - wavelengths[0])
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(shape)
    XYZ = np.trapezoid(np.asarray(spd, dtype=float)[:, None] * cmfs.values, wavelengths, axis=0)
    if XYZ[1] != 0:
        XYZ /= XYZ[1]
    denom = np.sum(XYZ)
    if denom == 0:
        return np.nan, np.nan
    return float(XYZ[0] / denom), float(XYZ[1] / denom)


def _normalize_max(arr: np.ndarray) -> np.ndarray:
    """Return arr divided by its maximum value, or a copy unchanged if max is 0."""
    m = float(np.max(arr))
    return arr.copy() if m == 0 else arr / m


# ── Daylight (D-series) ───────────────────────────────────────────────────────

def get_spd(Tc: float, S0: np.ndarray, S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """
    Compute the CIE D-series SPD for correlated colour temperature Tc (K).

    Uses the standard CIE formula with S0/S1/S2 basis functions sampled from
    the daylight_S012.xlsx spreadsheet.
    """
    if 4000 <= Tc <= 7000:
        xD = -4.6070e9 / Tc**3 + 2.9678e6 / Tc**2 + 9.911e1 / Tc + 0.244063
    elif 7000 < Tc <= 25000:
        xD = -2.0064e9 / Tc**3 + 1.9018e6 / Tc**2 + 2.4748e2 / Tc + 0.23704
    else:
        raise ValueError("Tc must be in [4000, 25000] K for CIE D-series.")
    yD = -3.000 * xD**2 + 2.870 * xD - 0.275
    denom = 0.0241 + 0.2562 * xD - 0.7341 * yD
    M1 = (-1.3515 - 1.7703 * xD + 5.9114 * yD) / denom
    M2 = (0.0300 - 31.4424 * xD + 30.0717 * yD) / denom
    return S0 + M1 * S1 + M2 * S2


def get_daylight_by_CCT(Tc: float) -> np.ndarray:
    """Return a D-series SPD for the given CCT, sliced to the 400–700 nm grid."""
    df = pd.read_excel(os.path.join(_ILLUMINANT_DIR, 'daylight_S012.xlsx'))
    return get_spd(Tc, df["S0"].to_numpy(), df["S1"].to_numpy(), df["S2"].to_numpy())[10:-13]


def get_all_day_light(
        df: pd.DataFrame,
        normalize: bool = True,
        dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Build standard D illuminants (D50, D55, D60, D65, D75, D93) from S0/S1/S2.

    The spreadsheet covers 300–830 nm; the [10:-13] slice trims to 400–700 nm.
    """
    name_list = ["D50", "D55", "D60", "D65", "D75", "D93"]
    S0 = df["S0"].to_numpy()
    S1 = df["S1"].to_numpy()
    S2 = df["S2"].to_numpy()
    wavelengths = df["Lamda"].to_numpy()[10:-13]

    spd_dict: dict[str, np.ndarray] = {}
    for name in name_list:
        Tc = int(name.replace("D", "")) * 100
        illum = get_spd(Tc, S0, S1, S2)[10:-13].astype(dtype)
        spd_dict[name] = _normalize_max(illum).astype(dtype) if normalize else illum

    return wavelengths, spd_dict


# ── Fluorescent ───────────────────────────────────────────────────────────────

def get_all_fluo(
        df: pd.DataFrame,
        normalize: bool = True,
        dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Load all fluorescent illuminants from the Fluorescents.xls sheet.

    The sheet is sampled at 5 nm; [::2] downsamples to 10 nm, then [2:-8]
    trims to 400–700 nm.
    """
    wavelengths = df["Lamda"].to_numpy()[::2][2:-8]
    spd_dict: dict[str, np.ndarray] = {}
    for name in list(df.keys())[1:]:  # skip 'Lamda'
        illum = df[name].to_numpy(dtype=dtype)[::2][2:-8]
        spd_dict[name] = _normalize_max(illum) if normalize else illum
    return wavelengths, spd_dict


# ── Illuminant A ──────────────────────────────────────────────────────────────

def get_A(
        df: pd.DataFrame,
        normalize: bool = True,
        dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Load CIE Illuminant A from the CIE_A.xlsx sheet.

    The sheet is sampled at 5 nm; [::2] downsamples to 10 nm, then [10:-13]
    trims to 400–700 nm.
    """
    wavelengths = df["Lamda"].to_numpy()[::2][10:-13]
    A = df["A"].to_numpy(dtype=dtype)[::2][10:-13]
    return wavelengths, {"A": _normalize_max(A) if normalize else A}


# ── LED illuminants ───────────────────────────────────────────────────────────

def _get_led_cols() -> list[str]:
    """Return column names for the LED illuminant CSV from its JSON metadata file."""
    with open(os.path.join(_ILLUMINANT_DIR, 'CIE_illum_LEDs.csv_metadata_v2.json'), 'r') as f:
        headers = json.load(f)['datatableInfo']['columnHeaders']
    return [h['title'] for h in headers]


def get_led_illuminants() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Load CIE LED illuminants from the CSV file.

    The CSV is sampled at 5 nm over 380–780 nm; [::2] downsamples to 10 nm,
    then [2:-8] trims to 400–700 nm.
    """
    cols = _get_led_cols()
    with open(os.path.join(_ILLUMINANT_DIR, 'CIE_illum_LEDs.csv'), 'r') as f:
        data = np.array(list(csv.reader(f)))
    data = data[::2][2:-8]
    wavelengths = data[:, 0].astype(np.int32)
    illum_dict = {name: data[:, idx + 1].astype(np.float32) for idx, name in enumerate(cols[1:])}
    return wavelengths, illum_dict


# ── Unified loader ────────────────────────────────────────────────────────────

def load_illuminants(
        illuminant_folder: str = _ILLUMINANT_DIR,
        include: Iterable[str] | None = ("daylight", "fluorescent", "A"),
        normalize: bool = True,
        dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Load and merge illuminant SPDs from multiple families into a single dict.

    Args:
        illuminant_folder: Directory containing the raw illuminant data files.
        include:           Which families to load: any subset of
                           {'daylight', 'fluorescent', 'A', 'led'}.
                           None loads all four.
        normalize:         If True, each SPD is normalised to max=1.
        dtype:             NumPy dtype for the SPD arrays.

    Returns:
        wavelengths: (31,) array for the 400–700 nm / 10 nm grid.
        illum_dict:  Dict mapping illuminant name → (31,) SPD array.

    Raises:
        ValueError: If wavelength grids across families do not match, or if
                    nothing is loaded (empty include set).
    """
    if include is None:
        include = ("daylight", "fluorescent", "A", "led")
    include = set(include)

    illum_dict: dict[str, np.ndarray] = {}
    wavelengths_ref: np.ndarray | None = None

    def _check_wavelengths(w: np.ndarray, label: str) -> None:
        nonlocal wavelengths_ref
        if wavelengths_ref is None:
            wavelengths_ref = w
        elif not np.array_equal(wavelengths_ref, w):
            raise ValueError(
                f"Wavelength grid mismatch for '{label}'.\n"
                f"Expected: {wavelengths_ref}\n"
                f"Got:      {w}"
            )

    if "daylight" in include:
        df = pd.read_excel(os.path.join(illuminant_folder, "daylight_S012.xlsx"))
        w, d = get_all_day_light(df, normalize=normalize, dtype=dtype)
        _check_wavelengths(w, "daylight")
        illum_dict.update(d)

    if "fluorescent" in include:
        df = pd.read_excel(os.path.join(illuminant_folder, "Fluorescents.xls"), skiprows=1)
        w, d = get_all_fluo(df, normalize=normalize, dtype=dtype)
        _check_wavelengths(w, "fluorescent")
        illum_dict.update(d)

    if "A" in include:
        df = pd.read_excel(os.path.join(illuminant_folder, "CIE_A.xlsx"))
        w, d = get_A(df, normalize=normalize, dtype=dtype)
        _check_wavelengths(w, "A")
        illum_dict.update(d)

    if "led" in include:
        w, d = get_led_illuminants()
        _check_wavelengths(w, "led")
        illum_dict.update(d)

    if wavelengths_ref is None:
        raise ValueError("Nothing loaded — check the 'include' argument and file paths.")

    return wavelengths_ref, illum_dict


if __name__ == '__main__':
    print("ILL_TEST_SET:", ILL_TEST_SET)
    print("ILL_TRAIN_SET:", ILL_TRAIN_SET)
