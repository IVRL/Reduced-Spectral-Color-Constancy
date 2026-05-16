"""
Dataset access helpers for the KAUST hyperspectral reflectance dataset.

Provides loaders for the train/test image+mask splits and an iterator over
precomputed per-illuminant radiance pixel files (produced by precompute_radiances.py).
"""

import os
import numpy as np

_DATASET_PATH = 'datasets'
RADIANCE_DIR_FMT = os.path.join(_DATASET_PATH, 'radiances_ds{downscale}')
RADIANCE_FILE_FMT = 'train_radiance_{name}.npy'
_DEFAULT_RADIANCE_DOWNSCALE = 4


def get_kaust_image_mask(split: str, downscale: int = 4):
    """
    Load a saved KAUST dataset split from disk.

    Args:
        split:     'train' or 'test'.
        downscale: Downscale factor the dataset was generated with.

    Returns:
        X: (N, H, W, 31) float32 reflectance array.
        M: (N, H, W) uint8 foreground mask array.
    """
    if split not in ('train', 'test'):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    X = np.load(os.path.join(_DATASET_PATH, f"X_{split}_ds{downscale}.npy"))
    M = np.load(os.path.join(_DATASET_PATH, f"M_{split}_ds{downscale}.npy"))
    return X, M


def iterate_radiance_pixels(illuminant_name: str, batch_size: int = 1_000_000, downscale: int = _DEFAULT_RADIANCE_DOWNSCALE):
    """
    Yield batches of precomputed L1-normalised radiance pixels for one illuminant.

    Files are read with mmap_mode='r' to avoid loading the full array into memory.

    Args:
        illuminant_name: Name of the illuminant (must match a precomputed file).
        batch_size:      Number of pixels per yielded batch.
        downscale:       Must match the downscale used in precompute_radiances.py.

    Yields:
        (B, 31) float32 array of radiance pixels.
    """
    path = os.path.join(
        RADIANCE_DIR_FMT.format(downscale=downscale),
        RADIANCE_FILE_FMT.format(name=illuminant_name)
    )
    rad = np.load(path, mmap_mode='r')
    for s in range(0, rad.shape[0], batch_size):
        yield np.array(rad[s:min(s + batch_size, rad.shape[0])], dtype=np.float32)


def kaust_image_mask_generator(split: str, downscale: int = 4):
    """
    Yield (image, mask) pairs one at a time for the given split.

    Args:
        split:     'train' or 'test'.
        downscale: Downscale factor the dataset was generated with.

    Yields:
        image: (H, W, 31) float32 reflectance array.
        mask:  (H, W) uint8 foreground mask.
    """
    X, M = get_kaust_image_mask(split, downscale)
    for i in range(X.shape[0]):
        yield X[i], M[i]
