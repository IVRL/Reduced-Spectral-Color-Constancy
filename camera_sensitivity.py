"""
Loads RGB camera spectral sensitivity functions from the CamSpec database.

The database covers 28 cameras sampled at 400–720 nm (10 nm steps, 33 bands).
We strip the last two bands to align with the project's 400–700 nm / 31-band grid.

Source: https://spectraldb.com (CamSpec database, camspec_database.txt)
"""

import os
from functools import cache

import numpy as np

_DEFAULT_CAMSPEC_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data_raw", "camera_sensitivities", "camspec_database.txt",
)


@cache
def get_camera_sensitivities(file_path: str = _DEFAULT_CAMSPEC_PATH) -> dict[str, np.ndarray]:
    """
    Parse the CamSpec database and return all camera sensitivity functions.

    The file format is groups of 4 lines per camera:
        line 0 — camera name
        line 1 — R channel (33 values)
        line 2 — G channel (33 values)
        line 3 — B channel (33 values)

    Returns a dict mapping camera name → (31, 3) float32 array,
    where rows are wavelengths (400–700 nm) and columns are R, G, B.
    Results are cached after the first load.
    """
    cam_sensitivities = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()

    cam_name = ''
    for idx, line in enumerate(lines):
        line = line.rstrip('\n')
        if idx % 4 == 0:
            cam_name = line
            cam_sensitivities[cam_name] = np.zeros((3, 33), dtype=np.float32)
        else:
            cam_sensitivities[cam_name][idx % 4 - 1, :] = np.array(line.split(), dtype=np.float32)
        if idx % 4 == 3:
            # Strip bands 31–32 (710, 720 nm) to match the project's 400–700 nm grid,
            # then transpose to (31, 3) so the first axis is wavelength.
            cam_sensitivities[cam_name] = cam_sensitivities[cam_name][:, :-2].T

    return cam_sensitivities


def get_camera_sensitivity(model: str = 'Canon 300D') -> np.ndarray:
    """
    Return the sensitivity function for a single camera model.

    Args:
        model: Camera name as it appears in the CamSpec database.

    Returns:
        (31, 3) float32 array — rows are wavelengths (400–700 nm), columns are R, G, B.
    """
    return get_camera_sensitivities()[model]


if __name__ == '__main__':
    print(list(get_camera_sensitivities().keys()))
