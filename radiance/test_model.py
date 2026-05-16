import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from cie_color_matching_functions import cie_1931_2deg_cmfs
from illuminants import load_illuminants

RAD_DATASET_PATH  = 'datasets/real_radiance_data'


def test_model_on_radiance(
        model,
        illuminant_names: list[str],
        dtype=np.float32,
        threshold: bool = False,
        downscale: int = 2,
) -> dict:
    """
    Test a CBC model against the real captured radiance images.

    Unlike test_model(), no synthetic illumination is applied — the illuminant
    is already baked into the radiance images. Ground truth illuminant SPDs are
    the whiteboard-derived estimates from extract_radiance_illuminants.py.

    Two angular error variants are reported:
      - vs GT SPD      : predicted candidate vs the continuous whiteboard estimate
      - snapped        : predicted candidate vs the nearest candidate to the GT SPD
                         (directly comparable to test_model() results)

    Returns a metrics dict with the same keys as test_model() plus
    snapped_* variants.
    """
    _, illuminants = load_illuminants(include=('daylight', 'fluorescent', 'A', 'led'))
    cmfs = cie_1931_2deg_cmfs()

    # L2-normalised candidate matrix  (K, λ)
    cand = np.stack([illuminants[n] for n in illuminant_names], axis=0).astype(np.float32)
    cand_norm = cand / np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)

    # Whitepoint versions  (K, 3)
    cand_wp = cand_norm @ cmfs
    cand_wp /= np.maximum(np.linalg.norm(cand_wp, axis=1, keepdims=True), 1e-12)

    # Load radiance dataset
    X     = np.load(f'{RAD_DATASET_PATH}/X_test_ds{downscale}.npy')   # (N, H, W, 31)
    M     = np.load(f'{RAD_DATASET_PATH}/M_test_ds{downscale}.npy')   # (N, H, W)
    names = np.load(f'{RAD_DATASET_PATH}/names_ds{downscale}.npy')    # (N,)

    # Ground truth SPDs from whiteboard extraction
    gt_spds_raw = np.load(
        f'{RAD_DATASET_PATH}/radiance_illuminants.npy', allow_pickle=True
    ).item()   # dict { name -> (31,) L2-normalised }

    estimated_likelihoods = []
    gt_spd_list = []

    for image, mask, name in tqdm(
        zip(X, M, names),
        desc='Testing on radiance',
        colour='green',
        ncols=160,
        total=len(X),
    ):
        pixels = image.reshape(-1, image.shape[-1]).astype(np.float32)
        mask_flat = mask.reshape(-1).astype(bool)
        pixels = pixels[~mask_flat]

        # L1 normalise radiance (no synthetic illuminant applied)
        row_sums = pixels.sum(axis=1, keepdims=True)
        pixels_norm = np.divide(pixels, np.clip(row_sums, 1e-12, None), dtype=dtype)

        log_likelihoods = model.estimate_illuminant(pixels_norm, threshold)
        estimated_likelihoods.append(log_likelihoods)
        gt_spd_list.append(gt_spds_raw[name])   # already L2-normalised

    estimated_likelihoods = np.vstack(estimated_likelihoods)   # (N, K)
    predicted = np.argmax(estimated_likelihoods, axis=1)       # (N,)

    gt_spds = np.stack(gt_spd_list, axis=0).astype(np.float32)  # (N, 31), L2-normalised

    # Snap each GT SPD to the nearest candidate (for accuracy + snapped errors)
    cos_to_cands = gt_spds @ cand_norm.T    # (N, K)
    snapped_gt   = np.argmax(cos_to_cands, axis=1)   # (N,)

    accuracy = float(accuracy_score(snapped_gt, predicted))

    # ── Angular error helpers ──────────────────────────────────────────────
    def cos_sim_to_ang_err_deg(cos_sim):
        return np.degrees(np.arccos(np.clip(np.asarray(cos_sim, dtype=float), -1.0, 1.0)))

    def angular_error_stats(ang_err_deg):
        x = np.asarray(ang_err_deg, dtype=float)
        q1, q2, q3 = np.quantile(x, [0.25, 0.50, 0.75])
        return {
            'mean':         float(np.mean(x)),
            'median':       float(np.median(x)),
            'trimean':      float(0.25 * q1 + 0.5 * q2 + 0.25 * q3),
            'best25_mean':  float(np.mean(x[x <= q1])),
            'worst25_mean': float(np.mean(x[x >= q3])),
        }

    # ── vs continuous GT SPD ──────────────────────────────────────────────
    pred_spds = cand_norm[predicted]              # (N, 31)
    cos_sim   = np.sum(pred_spds * gt_spds, axis=1)

    gt_wp   = gt_spds @ cmfs                      # (N, 3)
    gt_wp  /= np.maximum(np.linalg.norm(gt_wp, axis=1, keepdims=True), 1e-12)
    pred_wp = cand_wp[predicted]                  # (N, 3)
    cos_sim_wp = np.sum(pred_wp * gt_wp, axis=1)

    # ── vs snapped GT candidate ───────────────────────────────────────────
    snapped_spds = cand_norm[snapped_gt]          # (N, 31)
    cos_sim_snapped = np.sum(pred_spds * snapped_spds, axis=1)

    snapped_wp   = cand_wp[snapped_gt]            # (N, 3)
    cos_sim_snapped_wp = np.sum(pred_wp * snapped_wp, axis=1)

    s    = angular_error_stats(cos_sim_to_ang_err_deg(cos_sim))
    s_wp = angular_error_stats(cos_sim_to_ang_err_deg(cos_sim_wp))
    sn   = angular_error_stats(cos_sim_to_ang_err_deg(cos_sim_snapped))
    sn_wp= angular_error_stats(cos_sim_to_ang_err_deg(cos_sim_snapped_wp))

    return {
        'accuracy':                    accuracy,

        'mean_ang_err':                s['mean'],
        'median_ang_err':              s['median'],
        'trimean_ang_err':             s['trimean'],
        'best25_mean_ang_err':         s['best25_mean'],
        'worst25_ang_err':             s['worst25_mean'],

        'mean_ang_err_wp':             s_wp['mean'],
        'median_ang_err_wp':           s_wp['median'],
        'trimean_ang_err_wp':          s_wp['trimean'],
        'best25_mean_ang_err_wp':      s_wp['best25_mean'],
        'worst25_ang_err_wp':          s_wp['worst25_mean'],

        'mean_snapped_ang_err':        sn['mean'],
        'median_snapped_ang_err':      sn['median'],
        'trimean_snapped_ang_err':     sn['trimean'],
        'best25_mean_snapped_ang_err': sn['best25_mean'],
        'worst25_snapped_ang_err':     sn['worst25_mean'],

        'mean_snapped_ang_err_wp':        sn_wp['mean'],
        'median_snapped_ang_err_wp':      sn_wp['median'],
        'trimean_snapped_ang_err_wp':     sn_wp['trimean'],
        'best25_mean_snapped_ang_err_wp': sn_wp['best25_mean'],
        'worst25_snapped_ang_err_wp':     sn_wp['worst25_mean'],
    }
