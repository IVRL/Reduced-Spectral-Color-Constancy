"""
Generate train/test .npy datasets from the raw KAUST hyperspectral .h5 files.

Loads each .h5 file, extracts the first 31 spectral bands, normalises to [0, 1],
applies spatial downscaling, pairs with its foreground mask, then saves
train and test splits as separate .npy arrays.

Run as a script:

    python generate_datasets.py --downscale 2
"""

import os

import numpy as np
import h5py
import cv2
from tqdm import tqdm

_KAUST_H5_PATH    = 'data_raw/kaust/h5'
_KAUST_MASKS_PATH = 'data_raw/kaust/masks'
_DATASET_SAVE_PATH = 'datasets/'


def verify_data_exists():
    """Raise FileNotFoundError if any .h5 file is missing a corresponding mask."""
    h5_files = sorted(f for f in os.listdir(_KAUST_H5_PATH) if f.endswith('.h5'))
    if not h5_files:
        raise FileNotFoundError(
            f"No .h5 files found in {_KAUST_H5_PATH!r}. "
            "Download the dataset and place the .h5 files there."
        )

    mask_stems = {f[:-9] for f in os.listdir(_KAUST_MASKS_PATH) if f.endswith('_mask.png')}
    missing = [f for f in h5_files if f[:-3] not in mask_stems]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} h5 file(s) have no corresponding mask in {_KAUST_MASKS_PATH!r}:\n"
            + "\n".join(missing)
        )


def load_h5_file(file_path: str) -> dict:
    """Load all datasets from an .h5 file into a dict of numpy arrays."""
    with h5py.File(file_path, 'r') as f:
        return {key: f[key][()] for key in f.keys()}


def create_kaust_image_mask_datasets(
        kaust_h5_path: str = _KAUST_H5_PATH,
        kaust_mask_path: str = _KAUST_MASKS_PATH,
        save_path: str = _DATASET_SAVE_PATH,
        downscale: int = 1,
        shuffle: bool = True,
        seed: int = 42,
        split_idx: int = (2 * 409) // 3,  # 2/3 train split; 409 = total KAUST images
        suppress_tqdm: bool = False,
) -> None:
    """
    Process raw KAUST .h5 files into train/test .npy arrays.

    Each image is normalised to [0, 1] per-image, optionally downscaled
    with nearest-neighbour interpolation, and paired with its foreground mask.
    Files are shuffled with a fixed seed before splitting so the split is
    deterministic but not ordered by acquisition.

    Output files (written to save_path):
        X_train_ds{downscale}.npy  — (N_train, H, W, 31) float32
        M_train_ds{downscale}.npy  — (N_train, H, W) uint8  foreground mask
        X_test_ds{downscale}.npy
        M_test_ds{downscale}.npy
    """
    verify_data_exists()

    os.makedirs(save_path, exist_ok=True)

    h5_files = sorted(f for f in os.listdir(kaust_h5_path) if f.endswith('.h5'))
    indices = np.arange(len(h5_files))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    train_indices = indices[:split_idx]
    test_indices  = indices[split_idx:]

    def load_split(split_indices):
        X, M = [], []
        for idx in tqdm(split_indices, disable=suppress_tqdm):
            fname = h5_files[idx]
            h5 = load_h5_file(os.path.join(kaust_h5_path, fname))
            data = h5['img\\'].transpose(1, 2, 0)[..., :31].astype(np.float32)
            data /= data.max()

            mask_path = os.path.join(kaust_mask_path, f"{fname[:-3]}_mask.png")
            mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)

            if downscale != 1:
                H, W = data.shape[:2]
                new_size = (W // downscale, H // downscale)
                data = cv2.resize(data, new_size, interpolation=cv2.INTER_NEAREST)
                mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)

            X.append(data)
            M.append(mask)
        return np.stack(X), np.stack(M)

    tqdm.write(f"Loading and processing {len(train_indices)} training samples...")
    X_train, M_train = load_split(train_indices)
    tqdm.write(f"Loading and processing {len(test_indices)} test samples...")
    X_test, M_test = load_split(test_indices)

    np.save(os.path.join(save_path, f"X_train_ds{downscale}.npy"), X_train)
    np.save(os.path.join(save_path, f"M_train_ds{downscale}.npy"), M_train)
    np.save(os.path.join(save_path, f"X_test_ds{downscale}.npy"),  X_test)
    np.save(os.path.join(save_path, f"M_test_ds{downscale}.npy"),  M_test)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate KAUST train/test dataset .npy files.")
    parser.add_argument('--downscale', type=int, default=2,
                        help="Spatial downscale factor (default: 2)")
    args = parser.parse_args()
    create_kaust_image_mask_datasets(downscale=args.downscale)
