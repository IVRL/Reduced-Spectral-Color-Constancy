"""
Hyperspectral image rendering utilities.

Converts spectral image arrays to XYZ, sRGB, or CIE Lab via colour-science.
The 10 nm wavelength step is folded into the CMF integration as a scale factor.
"""

import numpy as np
import colour
from enum import Enum

from illuminants import load_illuminants


class CSpace(Enum):
    XYZ      = 'XYZ'
    sRGB     = 'sRGB'
    Lab      = 'Lab'
    SPECTRUM = 'spectrum'


def cie_1931_2deg_cmfs(wavelengths_nm: np.ndarray) -> np.ndarray:
    """
    Return CIE 1931 2° CMFs sampled at the given wavelengths.

    Args:
        wavelengths_nm: 1D array of evenly spaced wavelengths in nm.

    Returns:
        (N, 3) float32 array of X̄, Ȳ, Z̄ values.
    """
    msds = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    msds_i = msds.copy().align(
        colour.SpectralShape(wavelengths_nm[0], wavelengths_nm[-1], wavelengths_nm[1] - wavelengths_nm[0])
    )
    return msds_i.values.astype(np.float32)


def convert_spectral_to_cspace(spectral_image: np.ndarray, cspace: CSpace) -> np.ndarray:
    """
    Convert a spectral image to the requested colour space.

    Args:
        spectral_image: (N, λ) or (H, W, λ) float array of spectral values.
        cspace:         Target colour space (XYZ, sRGB, or Lab).

    Returns:
        Array in the requested colour space, same spatial shape with 3 channels.
    """
    wavelengths, _ = load_illuminants()
    cmfs = cie_1931_2deg_cmfs(wavelengths)

    # Integrate spectra against CMFs; multiply by 10 for the 10 nm wavelength step.
    spec_axis = 1 if spectral_image.ndim == 2 else 2
    xyz = np.tensordot(spectral_image, cmfs, axes=([spec_axis], [0])) * 10.0

    # Normalize so the maximum luminance (Y) = 1.
    y_max = xyz[..., 1].max()
    if y_max > 0:
        xyz /= y_max

    if cspace == CSpace.XYZ:
        return xyz
    if cspace == CSpace.Lab:
        return colour.XYZ_to_Lab(
            xyz,
            illuminant=colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"],
        )
    if cspace == CSpace.sRGB:
        return xyz_to_srgb_image(xyz)

    raise ValueError(f"Unsupported colour space: {cspace!r}")


def xyz_to_srgb_image(xyz_image: np.ndarray, clip: bool = True) -> np.ndarray:
    """
    Convert a D65-referenced XYZ image to sRGB.

    Args:
        xyz_image: (H, W, 3) or (N, 3) float array in XYZ.
        clip:      If True, clamp output to [0, 1].

    Returns:
        Array of the same shape in sRGB, float64.
    """
    srgb = colour.XYZ_to_sRGB(
        xyz_image.astype(np.float64),
        illuminant=colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"],
    )
    return np.clip(srgb, 0.0, 1.0) if clip else srgb
