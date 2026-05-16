"""
Fit and save pixel-space projection models (PCA, NMF, LDA) on KAUST training data.

Each function fits a projector on spectral chromaticities (reflectance × illuminant SPD,
L1-normalised per pixel) computed from the KAUST training set under a given set of
illuminants, then optionally saves the fitted model to disk with joblib.

Run as a script to fit individual models or all three for n_components 1–6:

    python fit_projections.py all
    python fit_projections.py pca --n_components 3
"""

import os

import numpy as np
import joblib
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA, MiniBatchNMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from threadpoolctl import threadpool_limits

from illuminants import load_illuminants, ILL_TRAIN_SET
from kaust_dataset import kaust_image_mask_generator, get_kaust_image_mask


def _l1_norm_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L1-normalise each row of X in-place. Rows summing to < eps are left near-zero."""
    s = X.sum(axis=1, keepdims=True)
    np.divide(X, np.maximum(s, eps), out=X)
    return X


def _load_illuminants(illuminant_names: list[str]) -> tuple:
    """Return a tuple of (31,) float32 SPD arrays for the requested illuminant names."""
    _, illum_dict = load_illuminants(include=('daylight', 'fluorescent', 'A', 'led'))
    illum_list = []
    for name in illuminant_names:
        if name not in illum_dict:
            raise ValueError(f"Unknown illuminant: {name!r}")
        illum_list.append(np.asarray(illum_dict[name], dtype=np.float32).copy())
    return tuple(illum_list)


def _try_load(model_path: str, label: str):
    """Load a joblib model if it exists, otherwise return None."""
    try:
        model = joblib.load(model_path)
        print(f"Loaded existing {label} model from {model_path}. "
              f"Set recalculate=True or delete the file to refit.")
        return model
    except FileNotFoundError:
        print(f"No existing {label} model found at {model_path}, fitting...")
        return None


def get_kaust_pca_model(
        n_components: int,
        illuminant_names: list[str],
        downscale: int = 4,
        save_model: bool = True,
        save_path: str = "models/pca/",
        recalculate: bool = False,
) -> IncrementalPCA:
    """
    Fit an IncrementalPCA on spectral chromaticities from the KAUST training set.

    Chromaticity = reflectance × illuminant SPD, L1-normalised per pixel.
    Processes one image at a time to avoid loading the full dataset into memory.
    Masked (foreground) pixels are excluded.
    """
    suffix = f"_{n_components}comp_{'_'.join(illuminant_names)}_ds{downscale}"
    model_path = os.path.join(save_path, f"kaust_reflectances_pca{suffix}.joblib")

    if not recalculate:
        model = _try_load(model_path, "PCA")
        if model is not None:
            return model

    illum_list = _load_illuminants(illuminant_names)
    pca = IncrementalPCA(n_components=n_components)

    for X, M in tqdm(kaust_image_mask_generator('train', downscale=downscale),
                     desc="Fitting PCA"):
        X = X.reshape(-1, X.shape[-1]).astype(np.float32, copy=False)
        M = M.reshape(-1).astype(bool)
        kept = X[~M]

        for spd in illum_list:
            rad = (kept * spd).astype(np.float32, copy=False)
            _l1_norm_rows(rad)
            with threadpool_limits(limits=1):
                pca.partial_fit(rad)

    if save_model:
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(pca, model_path)
        print(f"Saved PCA model to {model_path}")

    return pca


def get_kaust_nmf_model(
        n_components: int,
        illuminant_names: list[str],
        downscale: int = 4,
        save_model: bool = True,
        save_path: str = "models/nmf/",
        recalculate: bool = False,
) -> MiniBatchNMF:
    """
    Fit a MiniBatchNMF on spectral chromaticities from the KAUST training set.

    Chromaticity = reflectance × illuminant SPD, L1-normalised per pixel.
    Loads the full training set into memory, computes chromaticities under each
    illuminant, shuffles, then fits. Masked (foreground) pixels are excluded.
    """
    suffix = f"_{n_components}comp_{'_'.join(illuminant_names)}_ds{downscale}"
    model_path = os.path.join(save_path, f"kaust_reflectances_nmf{suffix}.joblib")

    if not recalculate:
        model = _try_load(model_path, "NMF")
        if model is not None:
            return model

    illum_list = _load_illuminants(illuminant_names)

    X, M = get_kaust_image_mask('train', downscale=downscale)
    print(f"Loaded train set: shape={X.shape}, size={X.nbytes / 1e9:.2f} GB")
    X = X.reshape(-1, X.shape[-1]).astype(np.float32, copy=False)
    M = M.reshape(-1).astype(bool)
    kept = X[~M]
    del X

    n_pixels = kept.shape[0]
    n_illum = len(illum_list)
    lit = np.empty((n_pixels * n_illum, kept.shape[1]), dtype=np.float32)

    print("Computing chromaticities...")
    for idx, spd in enumerate(illum_list):
        rad = (kept * spd).astype(np.float32, copy=False)
        _l1_norm_rows(rad)
        lit[idx * n_pixels: (idx + 1) * n_pixels] = rad

    print("Shuffling...")
    np.random.shuffle(lit)

    print("Fitting NMF...")
    nmf = MiniBatchNMF(n_components=n_components, batch_size=1024, verbose=False)
    with threadpool_limits(limits=1):
        nmf.fit(lit)

    if save_model:
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(nmf, model_path)
        print(f"Saved NMF model to {model_path}")

    return nmf


def get_kaust_lda_model(
        n_components: int,
        illuminant_names: list[str],
        downscale: int = 4,
        save_model: bool = True,
        save_path: str = "models/lda/",
        recalculate: bool = False,
) -> LinearDiscriminantAnalysis:
    """
    Fit an LDA on spectral chromaticities from the KAUST training set.

    Chromaticity = reflectance × illuminant SPD, L1-normalised per pixel.
    Each illuminant is treated as a separate class, so the projection separates
    illuminant-induced chromaticity shifts. Masked (foreground) pixels are excluded.
    """
    suffix = f"_{n_components}comp_{'_'.join(illuminant_names)}_ds{downscale}"
    model_path = os.path.join(save_path, f"kaust_reflectances_lda{suffix}.joblib")

    if not recalculate:
        model = _try_load(model_path, "LDA")
        if model is not None:
            return model

    illum_list = _load_illuminants(illuminant_names)

    X, M = get_kaust_image_mask('train', downscale=downscale)
    print(f"Loaded train set: shape={X.shape}, size={X.nbytes / 1e9:.2f} GB")
    X = X.reshape(-1, X.shape[-1]).astype(np.float32, copy=False)
    M = M.reshape(-1).astype(bool)
    kept = X[~M]
    del X

    n_pixels = kept.shape[0]
    n_illum = len(illum_list)
    lit = np.empty((n_pixels * n_illum, kept.shape[1]), dtype=np.float32)
    labels = np.empty(n_pixels * n_illum, dtype=np.uint8)

    print("Computing chromaticities...")
    for idx, spd in enumerate(illum_list):
        rad = (kept * spd).astype(np.float32, copy=False)
        _l1_norm_rows(rad)
        lit[idx * n_pixels: (idx + 1) * n_pixels] = rad
        labels[idx * n_pixels: (idx + 1) * n_pixels] = idx

    print("Shuffling...")
    p = np.random.permutation(n_pixels * n_illum)
    lit = lit[p]
    labels = labels[p]

    print("Fitting LDA...")
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(lit, labels)

    if save_model:
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(lda, model_path)
        print(f"Saved LDA model to {model_path}")

    return lda


if __name__ == '__main__':
    import argparse

    _MODELS = {'pca': get_kaust_pca_model, 'nmf': get_kaust_nmf_model, 'lda': get_kaust_lda_model}

    parser = argparse.ArgumentParser(description="Fit and save projection model(s) on KAUST training data.")
    parser.add_argument('model', choices=list(_MODELS) + ['all'], help="Model type to fit, or 'all' to fit all methods for n_components 1-6.")
    parser.add_argument('--n_components', type=int, default=6)
    parser.add_argument('--illuminants', nargs='+', default=ILL_TRAIN_SET, metavar='ILL',
                        help="Illuminant names to use (default: ILL_TRAIN_SET)")
    parser.add_argument('--downscale', type=int, default=8)
    parser.add_argument('--recalculate', action='store_true', default=False)
    args = parser.parse_args()

    if args.model == 'all':
        for n in range(1, 7):
            for name, fn in _MODELS.items():
                print(f"\n--- Fitting {name.upper()} with {n} components ---")
                fn(
                    n_components=n,
                    illuminant_names=args.illuminants,
                    downscale=args.downscale,
                    save_model=True,
                    recalculate=args.recalculate,
                )
    else:
        _MODELS[args.model](
            n_components=args.n_components,
            illuminant_names=args.illuminants,
            downscale=args.downscale,
            save_model=True,
            recalculate=args.recalculate,
        )
