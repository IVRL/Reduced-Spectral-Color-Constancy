"""
CIE 1931 2° colour matching functions sampled on the project's wavelength grid.
"""

import colour
import numpy as np


def cie_1931_2deg_cmfs() -> np.ndarray:
    """
    Return the CIE 1931 2° CMFs sampled at 400–700 nm in 10 nm steps.

    Returns:
        (31, 3) float32 array — rows are wavelengths, columns are X̄, Ȳ, Z̄.
    """
    msds = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    msds_i = msds.copy().align(colour.SpectralShape(400, 700, 10))
    return msds_i.values.astype(np.float32)
