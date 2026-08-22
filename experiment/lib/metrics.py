"""
metrics.py - Evaluation metrics for anomaly detection.

Image-level:  AUROC, AP
Pixel-level:  AUROC, AP, AUPRO, AUPIMO
Deployment:   F1, Recall, FPR at threshold
"""
import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from .data import flat_subsample

EPS = 1e-8


def _trapz(y, x):
    fn = getattr(np, 'trapezoid', None)
    return fn(y, x) if fn is not None else np.trapz(y, x)


def has_both(y):
    u = np.unique(y)
    return (0 in u) and (1 in u)


# ---- image-level ----

def image_metrics(score, label):
    if not has_both(label):
        return dict(auroc=np.nan, ap=np.nan)
    return dict(
        auroc=roc_auc_score(label, score),
        ap=average_precision_score(label, score),
    )


# ---- pixel-level ----

def pixel_metrics(maps, masks):
    if masks.sum() == 0:
        return dict(auroc=np.nan, ap=np.nan)
    s, y = flat_subsample(maps, masks)
    if not has_both(y):
        return dict(auroc=np.nan, ap=np.nan)
    return dict(
        auroc=roc_auc_score(y, s),
        ap=average_precision_score(y, s),
    )


def aupro(maps, masks, n_th=80, fpr_limit=0.3, max_imgs=150, seed=0):
    """Per-Region Overlap under FPR limit."""
    masks = masks.astype(np.uint8)
    idx = np.where(masks.reshape(len(masks), -1).sum(1) > 0)[0]
    if len(idx) == 0:
        return np.nan
    if len(idx) > max_imgs:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, max_imgs, replace=False))

    amaps, m = maps[idx], masks[idx]
    inv = (1 - m).astype(bool)
    n_neg = inv.sum()

    regions = []
    for mm in m:
        lab, k = ndimage.label(mm)
        regions.append([(lab == r) for r in range(1, k + 1)])

    ths = np.linspace(amaps.min(), amaps.max(), n_th)
    pros, fprs = [], []
    for th in ths[::-1]:
        b = amaps >= th
        ov = []
        for bi, regs in zip(b, regions):
            for r in regs:
                ov.append((bi & r).sum() / (r.sum() + EPS))
        if not ov:
            continue
        pros.append(np.mean(ov))
        fprs.append((b & inv).sum() / (n_neg + EPS))

    fprs, pros = np.array(fprs), np.array(pros)
    keep = fprs <= fpr_limit
    if keep.sum() < 2:
        return np.nan
    x, y = fprs[keep], pros[keep]
    o = np.argsort(x)
    return float(_trapz(y[o], x[o]) / fpr_limit)


def aupimo(maps, masks, img_label, fpr_lb=1e-3, fpr_ub=1e-1, n=50,
           max_imgs=300, seed=0):
    """Simplified AUPIMO (Bertoldo et al., 2024)."""
    normal = maps[img_label == 0]
    if len(normal) == 0:
        return np.nan
    neg = normal.reshape(-1)
    fprs = np.logspace(np.log10(fpr_lb), np.log10(fpr_ub), n)
    ths = np.quantile(neg, 1 - fprs)

    aidx = [i for i in np.where(img_label == 1)[0] if masks[i].sum() > 0]
    if not aidx:
        return np.nan
    if len(aidx) > max_imgs:
        rng = np.random.default_rng(seed)
        aidx = list(np.sort(rng.choice(aidx, max_imgs, replace=False)))

    logf = np.log10(fprs)
    span = logf[-1] - logf[0]
    vals = []
    for i in aidx:
        mb = masks[i].astype(bool)
        mp = maps[i]
        tpr = np.array([(mp[mb] >= t).mean() for t in ths])
        vals.append(_trapz(tpr, logf) / span)
    return float(np.mean(vals))


# ---- deployment ----

def f1_at(score, label, th):
    return f1_score(label, (score >= th).astype(int), zero_division=0)


def f1_max(score, label, n=200):
    ths = np.quantile(score, np.linspace(0, 1, n))
    return float(max(f1_at(score, label, t) for t in ths))


def recall_at(score, label, th):
    pos = label == 1
    if pos.sum() == 0:
        return np.nan
    return float(((score >= th) & pos).sum() / pos.sum())


def fpr_at(score, label, th):
    neg = label == 0
    if neg.sum() == 0:
        return np.nan
    return float(((score >= th) & neg).sum() / neg.sum())


def recall_at_far(score, label, far):
    neg = score[label == 0]
    pos = score[label == 1]
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    t = np.quantile(neg, 1 - far)
    return float((pos >= t).mean())
