"""
calzsad.py - CalZSAD core: post-processing for CLIP-ZSAD deployment.

Components:
    1. Top-1% pixel signal (replace native image score)
    2. Robust per-class normalization (median + MAD)
    3. Conformal threshold selection (K normal images)
"""
import numpy as np

EPS = 1e-8


# ---- 1. Top-1% pixel signal ----

def top1pct_score(maps):
    """Per-image score = mean of top-1% highest pixels from anomaly map.

    More discriminative than the model's native image score because it
    captures localized anomalous regions without being dominated by noise.
    """
    flat = maps.reshape(len(maps), -1).astype(np.float32)
    thr = np.quantile(flat, 0.99, axis=1, keepdims=True)
    mask = flat >= thr
    return (flat * mask).sum(1) / np.clip(mask.sum(1), 1, None)


# ---- 2. Robust per-class normalization ----

def robust_normalize(scores, ref_scores):
    """Normalize scores using robust statistics from reference (normal) scores.

    Uses median as center and MAD_left (median absolute deviation of
    scores below median) as scale, multiplied by 1.4826 for consistency
    with Gaussian standard deviation.

    Parameters
    ----------
    scores : array-like
        Scores to normalize.
    ref_scores : array-like
        Reference normal scores used to estimate center and scale.

    Returns
    -------
    z : np.ndarray
        Normalized scores.
    """
    ref = np.asarray(ref_scores, dtype=np.float64)
    mu = float(np.median(ref))
    left = ref[ref <= mu]
    if len(left) < 2:
        sigma = float(np.std(ref)) + EPS
    else:
        sigma = 1.4826 * float(np.median(np.abs(left - mu))) + EPS
    return (np.asarray(scores, dtype=np.float64) - mu) / sigma


# ---- 3. Conformal threshold selection ----

def conformal_threshold(ref_scores, alpha=0.05):
    """Conformal upper threshold from K normal reference scores.

    Sets threshold at the m-th order statistic where
    m = min(ceil((K+1)*(1-alpha)), K), providing a finite-sample
    guarantee: P(score > threshold) <= alpha under exchangeability.

    Parameters
    ----------
    ref_scores : array-like
        Scores from K known-normal calibration images.
    alpha : float
        Target false positive rate (default 0.05 = 5%).

    Returns
    -------
    threshold : float
    """
    ref = np.sort(ref_scores)
    K = len(ref)
    m = min(int(np.ceil((K + 1) * (1 - alpha))), K)
    return float(ref[m - 1])


# ---- Combined pipeline ----

def calzsad_classify(scores_test, ref_normal_scores, alpha=0.05):
    """Full CalZSAD image-level classification pipeline.

    1. Normalize test scores using robust stats from reference normals
    2. Set conformal threshold from normalized reference normals
    3. Classify: anomalous if normalized score > threshold

    Parameters
    ----------
    scores_test : np.ndarray
        Raw scores for test images.
    ref_normal_scores : np.ndarray
        Raw scores for K known-normal calibration images.
    alpha : float
        Target FPR.

    Returns
    -------
    predictions : np.ndarray of int
        1 = anomalous, 0 = normal
    z_test : np.ndarray
        Normalized test scores
    threshold : float
        The conformal threshold used
    """
    z_ref = robust_normalize(ref_normal_scores, ref_normal_scores)
    z_test = robust_normalize(scores_test, ref_normal_scores)
    threshold = conformal_threshold(z_ref, alpha)
    predictions = (z_test > threshold).astype(int)
    return predictions, z_test, threshold


def pixel_threshold(normal_maps, beta=0.01):
    """Compute pixel-level threshold from normal calibration images.

    Takes the (1-beta) quantile of all pixels from K normal images.

    Parameters
    ----------
    normal_maps : np.ndarray, shape (K, H, W)
        Anomaly maps of K normal calibration images.
    beta : float
        Target pixel false alarm rate (default 1%).

    Returns
    -------
    threshold : float
    """
    all_pixels = normal_maps.reshape(-1).astype(np.float64)
    return float(np.quantile(all_pixels, 1 - beta))
