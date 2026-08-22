#!/usr/bin/env python3
"""
calzsad_threshold.py - Few-normal-shot CalZSAD thresholding.

For each dataset and class:
1. Sample K normal images as calibration set
2. Compute top-1% pixel signal for all images
3. Normalize per-class using robust statistics (median + MAD)
4. Set conformal threshold from K calibration scores
5. Classify all test images and report F1, Recall, FPR

Also computes pixel-level Dice at FAR=1%.

Usage:
    python scripts/calzsad_threshold.py --model aaclip --K 20 --seeds 8
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.data import list_datasets, load_classes
from lib.metrics import has_both, recall_at_far
from lib.calzsad import top1pct_score, robust_normalize, conformal_threshold, pixel_threshold

# Load config
_cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'paths.yaml')
with open(_cfg_path) as f:
    cfg = yaml.safe_load(f)

REPORT_DATASETS = ['BTAD', 'MPDD', 'MVTec', 'Brain', 'Liver', 'Retina']


def prf(pred, y):
    """Precision, Recall, F1, FPR from binary predictions."""
    pos, neg = y == 1, y == 0
    tp = int((pred & pos).sum())
    fp = int((pred & neg).sum())
    fn = int(((pred == 0) & pos).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    fpr = fp / max(int(neg.sum()), 1)
    return 100 * f1, 100 * rec, 100 * prec, 100 * fpr


def eval_image_level(classes, K, alpha, seeds):
    """Few-shot conformal classification over one dataset."""
    per_seed = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        all_pred, all_y = [], []

        for c in classes:
            score = top1pct_score(c['maps'])
            y = c['img_label']
            ni = np.where(y == 0)[0]

            if len(ni) < K + 5 or (y == 1).sum() < 1:
                continue

            ref_idx = rng.choice(ni, K, replace=False)
            ref_scores = score[ref_idx]

            z_all = robust_normalize(score, ref_scores)
            z_ref = robust_normalize(ref_scores, ref_scores)
            thr = conformal_threshold(z_ref, alpha)

            test_mask = np.ones(len(score), bool)
            test_mask[ref_idx] = False

            pred = (z_all[test_mask] > thr).astype(int)
            all_pred.append(pred)
            all_y.append(y[test_mask])

        if not all_pred:
            return None
        pred = np.concatenate(all_pred)
        y = np.concatenate(all_y)
        if not has_both(y):
            return None
        per_seed.append(prf(pred, y))

    arr = np.array(per_seed)
    return arr.mean(0), arr.std(0)


def eval_pixel_level(classes, K, beta, seeds):
    """Pixel-level Dice at FAR=beta using conformal pixel threshold."""
    dice_list = []
    fpr_list = []

    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        all_dice, all_fpr = [], []

        for c in classes:
            y = c['img_label']
            ni = np.where(y == 0)[0]
            if len(ni) < K + 5:
                continue

            ref_idx = rng.choice(ni, K, replace=False)
            normal_maps = c['maps'][ref_idx]
            thr = pixel_threshold(normal_maps, beta)

            test_mask = np.ones(len(y), bool)
            test_mask[ref_idx] = False

            test_maps = c['maps'][test_mask]
            test_gt = c['masks'][test_mask]

            pred = (test_maps >= thr).astype(np.uint8)

            # Dice per anomalous image
            anom_idx = np.where(c['img_label'][test_mask] == 1)[0]
            for i in anom_idx:
                gt = test_gt[i].astype(bool)
                pr = pred[i].astype(bool)
                if gt.sum() == 0:
                    continue
                inter = (gt & pr).sum()
                dice = 2 * inter / (gt.sum() + pr.sum() + 1e-8)
                all_dice.append(dice)

            # Pixel FPR on normal images
            norm_idx = np.where(c['img_label'][test_mask] == 0)[0]
            for i in norm_idx:
                fp = pred[i].sum()
                total = pred[i].size
                all_fpr.append(fp / total)

        if all_dice:
            dice_list.append(np.mean(all_dice))
        if all_fpr:
            fpr_list.append(np.mean(all_fpr))

    return (np.mean(dice_list) * 100 if dice_list else 0,
            np.mean(fpr_list) * 100 if fpr_list else 0)


def oracle_f1(classes, alpha):
    """Per-class oracle F1 at FPR = alpha using ground-truth labels."""
    all_pred, all_y = [], []
    for c in classes:
        score = top1pct_score(c['maps'])
        y = c['img_label']
        neg = score[y == 0]
        if len(neg) == 0:
            continue
        t = float(np.quantile(neg, 1 - alpha))
        all_pred.append((score >= t).astype(int))
        all_y.append(y)
    if not all_pred:
        return 0
    pred = np.concatenate(all_pred)
    y = np.concatenate(all_y)
    f1, _, _, _ = prf(pred, y)
    return f1


def main():
    parser = argparse.ArgumentParser(description='CalZSAD few-shot thresholding')
    parser.add_argument('--model', default='aaclip')
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Image-level FPR target')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='Pixel-level FAR target')
    args = parser.parse_args()

    dump_dir = os.path.join(cfg['dump_root'], args.model)
    epoch = cfg.get('epoch')
    available = list_datasets(dump_dir)

    rows = []
    for ds in REPORT_DATASETS:
        if ds not in available:
            print(f'  skip {ds}')
            continue

        classes = load_classes(dump_dir, ds, epoch=epoch)
        print(f'\n  Processing {ds} ({len(classes)} classes)...')

        # Image-level
        r = eval_image_level(classes, args.K, args.alpha, args.seeds)
        if r is None:
            print(f'  {ds}: insufficient data')
            continue
        m, sd = r
        of1 = oracle_f1(classes, args.alpha)

        # Pixel-level
        dice, px_fpr = eval_pixel_level(classes, args.K, args.beta,
                                         min(args.seeds, 3))

        rows.append(dict(
            dataset=ds, K=args.K,
            f1=round(m[0], 2), f1_std=round(sd[0], 2),
            recall=round(m[1], 2),
            fpr=round(m[3], 2), fpr_std=round(sd[3], 2),
            oracle_f1=round(of1, 2),
            dice_far1=round(dice, 2),
            px_fpr=round(px_fpr, 2),
        ))
        print(f'  {ds}: F1={m[0]:.1f} FPR={m[3]:.1f}% Oracle={of1:.1f} '
              f'Dice@1%={dice:.1f} pxFPR={px_fpr:.2f}%')

    df = pd.DataFrame(rows)
    out = os.path.join(cfg['results_dir'], f'calzsad_{args.model}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    print(f'\n{"=" * 60}')
    print(f'CalZSAD results (K={args.K}, alpha={args.alpha}, '
          f'beta={args.beta}, seeds={args.seeds})')
    print(df.to_string(index=False))
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
