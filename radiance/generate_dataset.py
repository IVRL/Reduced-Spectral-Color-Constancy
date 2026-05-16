"""
Processes the real radiance .mat files into the same .npy format as the
reflectance dataset (generate_datasets.py), for use as a separate test set.

Only images that have a corresponding mask are included.

Sensor constants derived from the full dataset:
  DARK_FLOOR = 239   (global minimum — sensor dark current offset)
  SENSOR_MAX = 4095  (2^12 - 1 — 12-bit sensor saturation)

Outputs (in datasets/real_radiance_data/):
  X_test_ds{n}.npy  — (N, H, W, 31) float32, normalised to sensor dynamic range
  M_test_ds{n}.npy  — (N, H, W)     uint8
  names_ds{n}.npy   — (N,)           str, scene names for illuminant lookup
"""

import os
import argparse
import numpy as np
import scipy.io
import cv2
from tqdm import tqdm

RAD_DIR        = os.path.join('data_raw', 'kaust', 'radiance')
MASK_DIR       = os.path.join('data_raw', 'kaust', 'masks')
SAVE_PATH      = os.path.join('datasets', 'real_radiance_data')
ILLUMINANTS_PATH = os.path.join(SAVE_PATH, 'radiance_illuminants.npy')
N_BANDS        = 31     # 400–700 nm, matching the reflectance pipeline

DARK_FLOOR = 239   # global min across all 409 images (sensor dark current)
SENSOR_MAX = 4095  # 2^12 - 1, 12-bit sensor saturation


def generate(downscale: int = 2) -> None:
    os.makedirs(SAVE_PATH, exist_ok=True)

    mat_names = sorted(f.replace('.mat', '') for f in os.listdir(RAD_DIR) if f.endswith('.mat'))

    # Use the illuminant estimates as the source of truth for valid images —
    # any scene that failed illuminant extraction (missing/empty mask) is excluded.
    illuminants = np.load(ILLUMINANTS_PATH, allow_pickle=True).item()
    valid_names = [n for n in mat_names if n in illuminants]

    print(f'Mat files found        : {len(mat_names)}')
    print(f'With illuminant GT     : {len(valid_names)}')
    print(f'Skipped (no valid GT)  : {len(mat_names) - len(valid_names)}')

    X, M = [], []

    for name in tqdm(valid_names, desc=f'Processing radiance ds{downscale}', ncols=120):
        hsi = scipy.io.loadmat(os.path.join(RAD_DIR, name + '.mat'))['hsi'].astype(np.float32)
        hsi = hsi[:, :, :N_BANDS]

        # Normalise using global sensor constants to preserve inter-scene brightness
        hsi = (hsi - DARK_FLOOR) / (SENSOR_MAX - DARK_FLOOR)
        hsi = np.clip(hsi, 0.0, 1.0)

        mask = (cv2.imread(
            os.path.join(MASK_DIR, name + '_mask.png'), cv2.IMREAD_GRAYSCALE
        ) > 0).astype(np.uint8)

        if downscale != 1:
            H, W = hsi.shape[:2]
            new_size = (W // downscale, H // downscale)
            hsi  = cv2.resize(hsi,  new_size, interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)

        X.append(hsi)
        M.append(mask)

    X_arr = np.stack(X)           # (N, H, W, 31)
    M_arr = np.stack(M)           # (N, H, W)
    names = np.array(valid_names) # (N,)

    np.save(os.path.join(SAVE_PATH, f'X_test_ds{downscale}.npy'), X_arr)
    np.save(os.path.join(SAVE_PATH, f'M_test_ds{downscale}.npy'), M_arr)
    np.save(os.path.join(SAVE_PATH, f'names_ds{downscale}.npy'),  names)

    print(f'\nSaved to {SAVE_PATH}/')
    print(f'  X_test_ds{downscale}.npy : {X_arr.shape}  {X_arr.dtype}')
    print(f'  M_test_ds{downscale}.npy : {M_arr.shape}  {M_arr.dtype}')
    print(f'  names_ds{downscale}.npy  : {names.shape}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate real radiance test dataset.')
    parser.add_argument('--downscale', type=int, default=2, help='Spatial downscale factor (default: 2)')
    args = parser.parse_args()
    generate(args.downscale)
