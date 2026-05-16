"""
Precompute per-illuminant radiance pixel arrays from the KAUST train set.

For each illuminant, every non-masked train pixel is multiplied by the
illuminant SPD and L1-normalized over wavelengths. Results are saved as
one .npy file per illuminant under datasets/radiances_ds{downscale}/.

Run:
    python3 precompute_radiances.py              # default downscale=4
    python3 precompute_radiances.py --downscale 2
"""
import os
import argparse
import numpy as np
from tqdm import tqdm

from illuminants import load_illuminants
from kaust_dataset import get_kaust_image_mask, RADIANCE_DIR_FMT, RADIANCE_FILE_FMT


def _l1_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L1-normalise each row of x in-place along the last axis."""
    s = x.sum(axis=-1, keepdims=True)
    return np.divide(x, np.maximum(s, eps), out=x)


def precompute_radiances(downscale: int = 4) -> None:
    """
    Compute and save per-illuminant radiance pixel files for the training set.

    For each illuminant, multiplies every unmasked train pixel by the illuminant
    SPD, L1-normalises, and saves the result to
    datasets/radiances_ds{downscale}/train_radiance_{name}.npy.

    Args:
        downscale: Spatial downscale factor — must match a dataset generated
                   by generate_datasets.py.
    """
    out_dir = RADIANCE_DIR_FMT.format(downscale=downscale)
    os.makedirs(out_dir, exist_ok=True)

    # Load and flatten train set, removing foreground-masked pixels
    X, M = get_kaust_image_mask('train', downscale=downscale)
    print(f"Loaded train set: shape={X.shape}, size={X.nbytes / 1e9:.2f} GB")
    X = X.reshape(-1, X.shape[-1]).astype(np.float32, copy=False)
    M = M.reshape(-1).astype(bool)
    pixels = X[~M]
    del X, M
    print(f"Kept {pixels.shape[0]:,} unmasked pixels")

    _, illum_dict = load_illuminants(include=('daylight', 'fluorescent', 'A', 'led'))

    for name, spd in tqdm(illum_dict.items(), desc="Illuminants"):
        spd = np.asarray(spd, dtype=np.float32)
        radiance = (pixels * spd).astype(np.float32, copy=False)
        _l1_normalize(radiance)
        out_path = os.path.join(out_dir, RADIANCE_FILE_FMT.format(name=name))
        np.save(out_path, radiance)
        tqdm.write(f"  Saved {name}: {out_path}  [{radiance.nbytes / 1e6:.1f} MB]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Precompute per-illuminant radiance arrays.")
    parser.add_argument('--downscale', type=int, default=4)
    args = parser.parse_args()
    precompute_radiances(downscale=args.downscale)
