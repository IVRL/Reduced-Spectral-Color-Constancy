"""
Experiment settings: enums and settings validation for the CBC pipeline.

DimRedMethod enumerates the supported dimensionality-reduction methods.
CameraRGBMode enumerates the output modes for the camera (CAM) projector.
validate_kaust_settings() fills in defaults and validates a settings dict.
"""

from enum import Enum

from illuminants import load_illuminants
from camera_sensitivity import get_camera_sensitivities


class DimRedMethod(Enum):
    NMF     = 'nmf'
    ILL_NMF = 'ill_nmf'
    PCA     = 'pca'
    ILL_PCA = 'ill_pca'
    LDA     = 'lda'
    CAM     = 'cam'
    RAND    = 'rand'


class CameraRGBMode(Enum):
    RGB   = 'rgb'
    UV    = 'uv'
    UV3   = 'uv3'
    LOGUV = 'loguv'
    RG    = 'rg'
    RG3   = 'rg3'


def validate_kaust_settings(settings: dict | None = None) -> dict:
    """
    Apply defaults and validate a settings dict for a CBC experiment.

    CAM mode overrides num_components (3 for RGB, 2 otherwise) and
    nulls out projection illuminant names.
    Non-CAM modes null out camera-specific fields.
    RAND mode is the only method that uses random_projection_seed.

    Args:
        settings: Partial or complete settings dict. May be None (all defaults).

    Returns:
        A new dict with all required keys present and values validated.

    Raises:
        ValueError: If any value fails validation.
    """
    all_illuminant_names = list(
        load_illuminants(include=('daylight', 'fluorescent', 'A', 'led'))[1].keys()
    )

    defaults = {
        "experiment_name":            "",
        "run_notes":                  "",
        "test_illuminant_names":      all_illuminant_names,
        "projection_illuminant_names": all_illuminant_names,
        "dim_red_method":             DimRedMethod.ILL_PCA,
        "num_components":             3,
        "camera_rgb_mode":            CameraRGBMode.RGB,
        "camera_model":               "Canon 300D",
        "random_projection_seed":     42,
        "threshold":                  False,
        "histogram_bin_count":        30,
        "snr_db":                     None,  # None = no noise; float = SNR in dB
    }

    s = {**defaults, **(settings or {})}

    if not isinstance(s["experiment_name"], str):
        raise ValueError("experiment_name must be a str")
    if not isinstance(s["run_notes"], str):
        raise ValueError("run_notes must be a str")
    if not isinstance(s["num_components"], int) or s["num_components"] <= 0:
        raise ValueError("num_components must be a positive integer")
    if not isinstance(s["dim_red_method"], DimRedMethod):
        raise ValueError("dim_red_method must be a DimRedMethod enum value")
    if not isinstance(s["threshold"], bool):
        raise ValueError("threshold must be bool")
    if not isinstance(s["histogram_bin_count"], int) or s["histogram_bin_count"] <= 1:
        raise ValueError("histogram_bin_count must be an integer > 1")

    if s["dim_red_method"] == DimRedMethod.CAM:
        if not isinstance(s["camera_rgb_mode"], CameraRGBMode):
            raise ValueError(
                "camera_rgb_mode must be a CameraRGBMode enum value when dim_red_method=CAM"
            )
        if s["camera_model"] not in get_camera_sensitivities():
            raise ValueError(
                f"camera_model '{s['camera_model']}' not found in camera sensitivity file"
            )
        s["num_components"] = 3 if s["camera_rgb_mode"] == CameraRGBMode.RGB else 2
        s["projection_illuminant_names"] = None
    else:
        s["camera_rgb_mode"] = None
        s["camera_model"] = None

    if s["dim_red_method"] != DimRedMethod.RAND:
        s["random_projection_seed"] = None

    return s
