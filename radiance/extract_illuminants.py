"""
Estimate the illuminant SPD for each radiance image from the whiteboard mask.

The whiteboard is a near-perfect diffuse reflector, so:
    radiance_whiteboard ≈ illuminant_SPD  (up to a global scale)

Only the brightest pixels within the mask are used (controlled by
BRIGHT_PERCENTILE): darker border pixels and any shadow contamination are
discarded before taking the per-band median.

Output: datasets/real_radiance_data/radiance_illuminants.npy
    A dict  { scene_name: np.ndarray (31,) }  of L2-normalised SPD estimates.
"""

import os
import numpy as np
import scipy.io
import cv2
from tqdm import tqdm

RAD_DIR           = os.path.join('data_raw', 'kaust', 'radiance')
MASK_DIR          = os.path.join('data_raw', 'kaust', 'masks')
OUT_PATH          = os.path.join('datasets', 'real_radiance_data', 'radiance_illuminants.npy')
N_BANDS           = 31     # 400–700 nm, matching our pipeline
BRIGHT_PERCENTILE = 50     # keep only pixels brighter than this percentile of mask pixels


def estimate_illuminant(name: str) -> np.ndarray | None:
    """
    Returns a L2-normalised illuminant SPD estimate (31,) for one scene,
    or None if the mask is empty.
    """
    rad_path  = os.path.join(RAD_DIR,  name + '.mat')
    mask_path = os.path.join(MASK_DIR, name + '_mask.png')

    hsi = scipy.io.loadmat(rad_path)['hsi'].astype(np.float32)   # (H, W, 34)
    hsi = hsi[:, :, :N_BANDS]                                      # (H, W, 31)

    # Subtract per-image dark floor (sensor offset)
    hsi -= hsi.min()

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(bool)  # (H, W)

    whiteboard_pixels = hsi[mask]   # (N_masked, 31)
    if len(whiteboard_pixels) == 0:
        return None

    # Keep only the brightest pixels — discard dark border/shadow contamination.
    # Brightness = sum across bands for each pixel.
    brightness = whiteboard_pixels.sum(axis=1)          # (N_masked,)
    threshold  = np.percentile(brightness, BRIGHT_PERCENTILE)
    bright_pixels = whiteboard_pixels[brightness >= threshold]

    # Per-band median of the bright subset
    spd = np.median(bright_pixels, axis=0)   # (31,)

    # L2-normalise so results are scale-invariant
    norm = np.linalg.norm(spd)
    if norm < 1e-12:
        return None
    return spd / norm


if __name__ == '__main__':
    os.makedirs('datasets/real_radiance_data', exist_ok=True)

    mat_names = sorted(f.replace('.mat', '') for f in os.listdir(RAD_DIR) if f.endswith('.mat'))

    illuminants = {}
    failed = []

    for name in tqdm(mat_names, desc='Extracting illuminants', ncols=120):
        spd = estimate_illuminant(name)
        if spd is not None:
            illuminants[name] = spd
        else:
            failed.append(name)

    np.save(OUT_PATH, illuminants)
    print(f'\nSaved {len(illuminants)} illuminant estimates to {OUT_PATH}')
    if failed:
        print(f'Failed (empty mask): {failed}')
