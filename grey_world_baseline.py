"""
Spectral grey-world illuminant estimation baseline.

The grey-world assumption: the mean spectrum of an image's pixels estimates
the scene illuminant. The mean is L2-normalised and snapped to the nearest
candidate illuminant by cosine similarity.

Run as a script to evaluate on the synthetic KAUST test set or the real
radiance dataset:

    python grey_world_baseline.py
    python grey_world_baseline.py --radiance
"""

import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from scipy.special import softmax
from tqdm import tqdm

from cie_color_matching_functions import cie_1931_2deg_cmfs
from illuminants import load_illuminants, ILL_TEST_SET
from kaust_dataset import kaust_image_mask_generator, get_kaust_image_mask


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


def test_grey_world(illuminant_names: list[str], downscale: int = 2) -> dict:
    """
    Spectral grey-world baseline on the synthetic KAUST test set.

    For each (image, illuminant) pair: multiply reflectance by the illuminant
    SPD, take the mean spectrum, L2-normalise, and snap to the nearest candidate
    by cosine similarity. Returns a metrics dict matching test_model() format.
    """
    _, illuminants = load_illuminants(include=('daylight', 'fluorescent', 'A', 'led'))
    cmfs = cie_1931_2deg_cmfs()

    # L2-normalised candidate matrix  (K, λ)
    cand = np.stack([illuminants[n] for n in illuminant_names], axis=0).astype(np.float32)
    cand_norm = cand / np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)

    # Whitepoint (CIE XYZ) versions of candidates  (K, 3)
    cand_wp = cand_norm @ cmfs
    cand_wp /= np.maximum(np.linalg.norm(cand_wp, axis=1, keepdims=True), 1e-12)

    estimated_scores = []
    ground_truth = np.empty((0,), dtype=np.int32)

    X_test, _ = get_kaust_image_mask('test', downscale=downscale)
    total = X_test.shape[0]
    del X_test

    for image, mask in tqdm(
        kaust_image_mask_generator('test', downscale=downscale),
        desc='Grey-world baseline — test set',
        colour='green',
        ncols=160,
        total=total,
    ):
        image = image.reshape(-1, image.shape[-1]).astype(np.float32, copy=False)
        mask  = mask.reshape(-1).astype(bool)
        image = image[~mask]

        for ill_idx, name in enumerate(illuminant_names):
            radiance = (image * illuminants[name]).astype(np.float32)

            estimate = radiance.mean(axis=0)
            estimate_norm = estimate / max(float(np.linalg.norm(estimate)), 1e-12)

            estimated_scores.append(cand_norm @ estimate_norm)  # (K,)
            ground_truth = np.append(ground_truth, ill_idx)

    estimated_scores = np.vstack(estimated_scores)   # (N, K)
    predicted        = np.argmax(estimated_scores, axis=1)

    cos_sim    = np.sum(cand_norm[predicted] * cand_norm[ground_truth], axis=1)
    cos_sim_wp = np.sum(cand_wp[predicted]   * cand_wp[ground_truth],   axis=1)

    ang_stats    = _angular_error_stats(_cos_sim_to_ang_err_deg(cos_sim))
    ang_stats_wp = _angular_error_stats(_cos_sim_to_ang_err_deg(cos_sim_wp))

    accuracy = float(accuracy_score(ground_truth, predicted))

    gt_one_hot = np.zeros_like(estimated_scores)
    for i, idx in enumerate(ground_truth):
        gt_one_hot[i, idx] = 1
    cross_entropy = log_loss(gt_one_hot, softmax(estimated_scores, axis=1))

    return {
        'accuracy':               accuracy,
        'cross_entropy':          cross_entropy,
        'mean_ang_err':           ang_stats['mean'],
        'median_ang_err':         ang_stats['median'],
        'trimean_ang_err':        ang_stats['trimean'],
        'best25_mean_ang_err':    ang_stats['best25_mean'],
        'worst25_ang_err':        ang_stats['worst25_mean'],
        'mean_ang_err_wp':        ang_stats_wp['mean'],
        'median_ang_err_wp':      ang_stats_wp['median'],
        'trimean_ang_err_wp':     ang_stats_wp['trimean'],
        'best25_mean_ang_err_wp': ang_stats_wp['best25_mean'],
        'worst25_ang_err_wp':     ang_stats_wp['worst25_mean'],
    }


RAD_DATASET_PATH = 'datasets/real_radiance_data'


def test_grey_world_on_radiance(
        illuminant_names: list[str],
        downscale: int = 2,
) -> dict:
    """
    Spectral grey-world baseline on the real captured radiance dataset.

    No synthetic illumination is applied — the illuminant is already baked in.
    Ground truth SPDs come from the whiteboard extraction. Returns a metrics dict
    matching test_model_on_radiance() format, including both continuous (vs raw
    extracted GT SPD) and snapped (vs nearest candidate) angular errors.
    """
    _, illuminants = load_illuminants(include=('daylight', 'fluorescent', 'A', 'led'))
    cmfs = cie_1931_2deg_cmfs()

    cand = np.stack([illuminants[n] for n in illuminant_names], axis=0).astype(np.float32)
    cand_norm = cand / np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)

    cand_wp = cand_norm @ cmfs
    cand_wp /= np.maximum(np.linalg.norm(cand_wp, axis=1, keepdims=True), 1e-12)

    X     = np.load(f'{RAD_DATASET_PATH}/X_test_ds{downscale}.npy')
    M     = np.load(f'{RAD_DATASET_PATH}/M_test_ds{downscale}.npy')
    names = np.load(f'{RAD_DATASET_PATH}/names_ds{downscale}.npy')

    gt_spds_raw = np.load(
        f'{RAD_DATASET_PATH}/radiance_illuminants.npy', allow_pickle=True
    ).item()

    estimated_scores     = []
    estimates_continuous = []
    gt_spd_list          = []

    for image, mask, name in tqdm(
        zip(X, M, names),
        desc='Grey-world baseline — radiance test set',
        colour='green',
        ncols=160,
        total=len(X),
    ):
        pixels = image.reshape(-1, image.shape[-1]).astype(np.float32)
        pixels = pixels[~mask.reshape(-1).astype(bool)]

        estimate = pixels.mean(axis=0)
        estimate_norm = estimate / max(float(np.linalg.norm(estimate)), 1e-12)

        estimated_scores.append(cand_norm @ estimate_norm)
        estimates_continuous.append(estimate_norm)
        gt_spd_list.append(gt_spds_raw[name])

    estimated_scores = np.vstack(estimated_scores)       # (N, K)
    predicted        = np.argmax(estimated_scores, axis=1)
    est_cont         = np.vstack(estimates_continuous)   # (N, 31)

    gt_spds    = np.stack(gt_spd_list, axis=0).astype(np.float32)  # (N, 31)
    snapped_gt = np.argmax(gt_spds @ cand_norm.T, axis=1)          # nearest candidate index

    accuracy = float(accuracy_score(snapped_gt, predicted))

    pred_spds = cand_norm[predicted]
    gt_wp     = gt_spds @ cmfs
    gt_wp    /= np.maximum(np.linalg.norm(gt_wp, axis=1, keepdims=True), 1e-12)
    pred_wp   = cand_wp[predicted]

    est_cont_wp  = est_cont @ cmfs
    est_cont_wp /= np.maximum(np.linalg.norm(est_cont_wp, axis=1, keepdims=True), 1e-12)

    def _stats(a, b):
        return _angular_error_stats(_cos_sim_to_ang_err_deg(np.sum(a * b, axis=1)))

    s     = _stats(pred_spds,  gt_spds)
    s_wp  = _stats(pred_wp,    gt_wp)
    sn    = _stats(pred_spds,  cand_norm[snapped_gt])
    sn_wp = _stats(pred_wp,    cand_wp[snapped_gt])
    sc    = _stats(est_cont,   gt_spds)
    sc_wp = _stats(est_cont_wp, gt_wp)

    return {
        'accuracy':                           accuracy,

        # Snapped prediction vs continuous GT SPD
        'mean_ang_err':                       s['mean'],
        'median_ang_err':                     s['median'],
        'trimean_ang_err':                    s['trimean'],
        'best25_mean_ang_err':                s['best25_mean'],
        'worst25_ang_err':                    s['worst25_mean'],
        'mean_ang_err_wp':                    s_wp['mean'],
        'median_ang_err_wp':                  s_wp['median'],
        'trimean_ang_err_wp':                 s_wp['trimean'],
        'best25_mean_ang_err_wp':             s_wp['best25_mean'],
        'worst25_ang_err_wp':                 s_wp['worst25_mean'],

        # Snapped prediction vs snapped GT candidate
        'mean_snapped_ang_err':               sn['mean'],
        'median_snapped_ang_err':             sn['median'],
        'trimean_snapped_ang_err':            sn['trimean'],
        'best25_mean_snapped_ang_err':        sn['best25_mean'],
        'worst25_snapped_ang_err':            sn['worst25_mean'],
        'mean_snapped_ang_err_wp':            sn_wp['mean'],
        'median_snapped_ang_err_wp':          sn_wp['median'],
        'trimean_snapped_ang_err_wp':         sn_wp['trimean'],
        'best25_mean_snapped_ang_err_wp':     sn_wp['best25_mean'],
        'worst25_snapped_ang_err_wp':         sn_wp['worst25_mean'],

        # Continuous grey-world estimate vs continuous GT SPD (no snapping on either side)
        'mean_continuous_ang_err':            sc['mean'],
        'median_continuous_ang_err':          sc['median'],
        'trimean_continuous_ang_err':         sc['trimean'],
        'best25_mean_continuous_ang_err':     sc['best25_mean'],
        'worst25_continuous_ang_err':         sc['worst25_mean'],
        'mean_continuous_ang_err_wp':         sc_wp['mean'],
        'median_continuous_ang_err_wp':       sc_wp['median'],
        'trimean_continuous_ang_err_wp':      sc_wp['trimean'],
        'best25_mean_continuous_ang_err_wp':  sc_wp['best25_mean'],
        'worst25_continuous_ang_err_wp':      sc_wp['worst25_mean'],
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--radiance', action='store_true',
                        help='Evaluate on real radiance dataset instead of synthetic test set')
    args = parser.parse_args()

    if args.radiance:
        res = test_grey_world_on_radiance(ILL_TEST_SET)
        hdr = "Grey-world baseline — Radiance test set"
        rows = [
            ("Illuminant accuracy",        f"{res['accuracy']:.2%}"),
            ("mean_ang_err",               f"{res['mean_ang_err']:.4f}"),
            ("mean_ang_err_wp",            f"{res['mean_ang_err_wp']:.4f}"),
            ("mean_snapped_ang_err",       f"{res['mean_snapped_ang_err']:.4f}"),
            ("mean_snapped_ang_err_wp",    f"{res['mean_snapped_ang_err_wp']:.4f}"),
            ("mean_continuous_ang_err",    f"{res['mean_continuous_ang_err']:.4f}"),
            ("mean_continuous_ang_err_wp", f"{res['mean_continuous_ang_err_wp']:.4f}"),
        ]
    else:
        res = test_grey_world(ILL_TEST_SET)
        hdr = "Grey-world baseline — Test set"
        rows = [
            ("Illuminant accuracy", f"{res['accuracy']:.2%}"),
            ("mean_ang_err",        f"{res['mean_ang_err']:.4f}"),
            ("mean_ang_err_wp",     f"{res['mean_ang_err_wp']:.4f}"),
        ]

    w = max(len(k) for k, _ in rows)
    tqdm.write(f"\n{hdr}\n" + "\n".join(f"  {k:<{w}} : {v}" for k, v in rows) + "\n")
