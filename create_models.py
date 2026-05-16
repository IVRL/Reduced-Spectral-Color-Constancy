"""
Model factory for the CBC pipeline.

create_model() assembles a projector from the requested dimensionality-reduction
method, scans the training radiance pixels to determine global projection extrema,
then builds a log-probability histogram for each candidate illuminant.
"""

import numpy as np
from sklearn.decomposition import PCA as sklearn_pca
from sklearn.decomposition import NMF as sklearn_nmf
from tqdm import tqdm

from cbc_model import Model, SimpleCBC
from experiment_settings import DimRedMethod, validate_kaust_settings
from camera_sensitivity import get_camera_sensitivity
from fit_projections import get_kaust_pca_model, get_kaust_nmf_model, get_kaust_lda_model
from projector import SensorProjector, SklearnProjector, RandomProjector
from illuminants import load_illuminants
from kaust_dataset import iterate_radiance_pixels


def create_model(settings: dict, verbose: bool = True, tqdm_write: bool = True) -> Model:
    """
    Build a SimpleCBC model from a validated settings dict.

    Two-phase construction:
      1. Scan all training radiance pixels to find global min/max in projection
         space — these become the histogram bin edges.
      2. For each candidate illuminant, accumulate a pixel histogram in that
         space, apply Laplace smoothing, and store as log-probabilities.

    Args:
        settings:    Experiment settings dict.
        verbose:     Whether to print progress messages.
        tqdm_write:  If True, use tqdm.write instead of print (safe inside tqdm loops).

    Returns:
        A fitted SimpleCBC model ready for illuminant estimation.
    """

    def _print(s: str):
        if verbose:
            tqdm.write(s) if tqdm_write else print(s)

    settings = validate_kaust_settings(settings)
    projection_illuminant_names = settings["projection_illuminant_names"]
    test_illuminant_names       = settings["test_illuminant_names"]
    num_components              = settings["num_components"]
    method                      = settings["dim_red_method"]
    camera_rgb_mode             = settings["camera_rgb_mode"]
    camera_model                = settings["camera_model"]
    num_bins                    = settings["histogram_bin_count"]
    random_projection_seed      = settings["random_projection_seed"]

    _, illuminants = load_illuminants(include=('daylight', 'fluorescent', 'A', 'led'))

    # ── Build projector ───────────────────────────────────────────────────────
    if method == DimRedMethod.CAM:
        camera_sensitivities = get_camera_sensitivity(model=camera_model)
        projector = SensorProjector(sensitivities=camera_sensitivities, mode=camera_rgb_mode)

    elif method == DimRedMethod.ILL_PCA:
        # Fit PCA on L1-normalised illuminant SPDs rather than on image pixels.
        illum_spectra = np.vstack([s / max(s.sum(), 1e-12) for s in illuminants.values()])
        pca = sklearn_pca(n_components=num_components).fit(illum_spectra)
        projector = SklearnProjector(skl_projection=pca)

    elif method == DimRedMethod.ILL_NMF:
        illum_spectra = np.vstack([s / max(s.sum(), 1e-12) for s in illuminants.values()])
        nmf = sklearn_nmf(n_components=num_components, max_iter=5000).fit(illum_spectra)
        projector = SklearnProjector(skl_projection=nmf)

    elif method == DimRedMethod.PCA:
        pca = get_kaust_pca_model(n_components=num_components, illuminant_names=projection_illuminant_names, save_model=True, save_path='models/pca/', recalculate=False)
        projector = SklearnProjector(skl_projection=pca)

    elif method == DimRedMethod.NMF:
        nmf = get_kaust_nmf_model(n_components=num_components, illuminant_names=projection_illuminant_names, save_model=True, save_path='models/nmf/', recalculate=False)
        projector = SklearnProjector(skl_projection=nmf)

    elif method == DimRedMethod.LDA:
        lda = get_kaust_lda_model(n_components=num_components, illuminant_names=projection_illuminant_names)
        projector = SklearnProjector(skl_projection=lda)

    elif method == DimRedMethod.RAND:
        projector = RandomProjector(n_components=num_components, seed=random_projection_seed)

    else:
        raise ValueError(f"Unknown method: {method}")

    D = projector.n_components
    BATCH_PX = 600_000

    # ── Phase 1: find global projection extrema for histogram bin edges ───────
    proj_mins = np.full(D, np.inf,  dtype=np.float64)
    proj_maxs = np.full(D, -np.inf, dtype=np.float64)

    _print("Finding global projection extrema")
    for name in tqdm(test_illuminant_names, desc='Finding extrema', colour='green', ncols=160, disable=not verbose):
        for px in iterate_radiance_pixels(name, batch_size=BATCH_PX):
            Z = projector.transform(px)[:, :D]
            proj_mins = np.minimum(proj_mins, Z.min(axis=0).astype(np.float64))
            proj_maxs = np.maximum(proj_maxs, Z.max(axis=0).astype(np.float64))

    edges = [np.linspace(proj_mins[d], proj_maxs[d], num_bins + 1, dtype=np.float32) for d in range(D)]

    # ── Phase 2: build per-illuminant log-probability histograms ─────────────
    _print("Building illuminant histograms")
    M = len(test_illuminant_names)
    combined_hists = np.empty((M, num_bins ** D), dtype=np.float32)

    for idx, name in tqdm(enumerate(test_illuminant_names), total=M, desc='Building histograms', colour='green', ncols=160, unit='illuminant', disable=not verbose):
        H = np.zeros((num_bins,) * D, dtype=np.float32)
        for px in iterate_radiance_pixels(name, batch_size=BATCH_PX):
            Z = projector.transform(px)[:, :D]
            h, _ = np.histogramdd(Z, bins=edges)
            H += h.astype(np.float32, copy=False)

        # Laplace smoothing (+1) so no bin has zero probability,
        # then normalise and take log for efficient dot-product scoring.
        H += 1.0
        H /= H.sum(dtype=np.float64)
        combined_hists[idx] = np.log(H.astype(np.float32)).ravel()

    return SimpleCBC({'projector': projector, 'combined_hists': combined_hists, 'edges': edges})
