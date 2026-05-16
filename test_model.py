"""
Evaluation of a CBC model on the synthetic KAUST test set.

For each (image, illuminant) pair in the test set, the image pixels are lit
under the illuminant, L1-normalized, and scored by the model. Metrics include
classification accuracy, cross-entropy, and spectral/whitepoint angular errors.
"""

import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from scipy.special import softmax
from tqdm import tqdm

from cie_color_matching_functions import cie_1931_2deg_cmfs
from illuminants import load_illuminants
from kaust_dataset import kaust_image_mask_generator, get_kaust_image_mask
from noise import add_snr_noise
from cbc_model import Model


def _cos_sim_to_ang_err_deg(cos_sim) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.asarray(cos_sim, dtype=float), -1.0, 1.0)))


def _angular_error_stats(ang_err_deg) -> dict:
    x = np.asarray(ang_err_deg, dtype=float)
    q1, q2, q3 = np.quantile(x, [0.25, 0.50, 0.75])
    return {
        'mean':         float(np.mean(x)),
        'median':       float(np.median(x)),
        'trimean':      float(0.25 * q1 + 0.5 * q2 + 0.25 * q3),
        'best25_mean':  float(np.mean(x[x <= q1])),
        'worst25_mean': float(np.mean(x[x >= q3])),
    }


def test_model(
        model: Model,
        illuminant_names: list[str],
        threshold: bool = False,
        downscale: int = 2,
        snr_db: float | None = None,
) -> dict:
    """
    Evaluate a CBC model on the KAUST synthetic test set.

    Each test image is lit under every candidate illuminant (reflectance ×
    L1-normalized illuminant SPD), then scored by the model. Ground truth is
    the illuminant used to generate the radiance.

    Args:
        model:            Fitted SimpleCBC model.
        illuminant_names: Candidate illuminant names to evaluate against.
        threshold:        If True, clip image histogram counts to 1 before scoring.
        downscale:        Spatial downscale factor of the test dataset.
        snr_db:           If set, add Gaussian noise at this SNR (dB) before scoring.

    Returns:
        Dict of metrics: accuracy, cross_entropy, and spectral/whitepoint
        angular error statistics (mean, median, trimean, best25, worst25).
    """
    _, illuminants = load_illuminants(include=('daylight', 'fluorescent', 'A', 'led'))
    cmfs = cie_1931_2deg_cmfs()
    rng  = np.random.default_rng(42)

    estimated_likelihoods    = []
    ground_truth_illuminants = np.empty((0,), dtype=np.int32)

    X_test, _ = get_kaust_image_mask('test', downscale=downscale)
    total = X_test.shape[0]
    del X_test

    for image, mask in tqdm(
        kaust_image_mask_generator("test", downscale=downscale),
        desc="Testing", colour='green', ncols=160, total=total,
    ):
        image = image.reshape(-1, image.shape[-1]).astype(np.float32, copy=False)
        image = image[~mask.reshape(-1).astype(bool)]

        for ill_idx, name in enumerate(illuminant_names):
            illum = illuminants[name]
            illum_norm = illum / max(illum.sum(), 1e-12)  # L1-normalize the illuminant SPD
            radiance = (image * illum_norm).astype(np.float32, copy=False)

            if snr_db is not None:
                radiance = add_snr_noise(radiance, snr_db, rng)

            # L1-normalize per pixel to get chromaticities
            rad_sums = radiance.sum(axis=1, keepdims=True)
            radiance_norm = np.divide(radiance, np.clip(rad_sums, 1e-12, None))

            estimated_likelihoods.append(model.estimate_illuminant(radiance_norm, threshold))
            ground_truth_illuminants = np.append(ground_truth_illuminants, ill_idx)

    estimated_likelihoods    = np.vstack(estimated_likelihoods)
    predicted_illuminants    = np.argmax(estimated_likelihoods, axis=1)

    accuracy      = float(accuracy_score(ground_truth_illuminants, predicted_illuminants))
    cross_entropy = log_loss(
        np.eye(len(illuminant_names))[ground_truth_illuminants],
        softmax(estimated_likelihoods, axis=1),
    )

    # L2-normalize candidate SPDs for cosine similarity
    Ill      = np.stack([illuminants[n] for n in illuminant_names]).astype(np.float32)
    Ill_norm = Ill / np.maximum(np.linalg.norm(Ill, axis=1, keepdims=True), 1e-12)

    Ill_wp   = Ill_norm @ cmfs
    Ill_wp  /= np.maximum(np.linalg.norm(Ill_wp, axis=1, keepdims=True), 1e-12)

    cos_sim    = np.sum(Ill_norm[predicted_illuminants] * Ill_norm[ground_truth_illuminants], axis=1)
    cos_sim_wp = np.sum(Ill_wp[predicted_illuminants]   * Ill_wp[ground_truth_illuminants],   axis=1)

    s    = _angular_error_stats(_cos_sim_to_ang_err_deg(cos_sim))
    s_wp = _angular_error_stats(_cos_sim_to_ang_err_deg(cos_sim_wp))

    return {
        'accuracy':               accuracy,
        'cross_entropy':          cross_entropy,
        'mean_ang_err':           s['mean'],
        'median_ang_err':         s['median'],
        'trimean_ang_err':        s['trimean'],
        'best25_mean_ang_err':    s['best25_mean'],
        'worst25_ang_err':        s['worst25_mean'],
        'mean_ang_err_wp':        s_wp['mean'],
        'median_ang_err_wp':      s_wp['median'],
        'trimean_ang_err_wp':     s_wp['trimean'],
        'best25_mean_ang_err_wp': s_wp['best25_mean'],
        'worst25_ang_err_wp':     s_wp['worst25_mean'],
    }
