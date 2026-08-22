#!/usr/bin/env python3
"""
evaluate.py - Compute ranking metrics on AA-CLIP dumps.

Reports per-dataset: img-AUROC, img-AP, px-AUROC, px-AP, AUPRO, AUPIMO.

Usage:
    python scripts/evaluate.py --model aaclip
"""
import os
import sys
import argparse
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.data import list_datasets, load_dataset
from lib.metrics import image_metrics, pixel_metrics, aupro, aupimo

# Load config
_cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'paths.yaml')
with open(_cfg_path) as f:
    cfg = yaml.safe_load(f)

# Only evaluate on the 6 report datasets
REPORT_DATASETS = ['BTAD', 'MPDD', 'MVTec', 'Brain', 'Liver', 'Retina']


def main():
    parser = argparse.ArgumentParser(description='Evaluate ranking metrics')
    parser.add_argument('--model', default='aaclip')
    args = parser.parse_args()

    dump_dir = os.path.join(cfg['dump_root'], args.model)
    epoch = cfg.get('epoch')
    available = list_datasets(dump_dir)

    rows = []
    for ds in REPORT_DATASETS:
        if ds not in available:
            print(f'  skip {ds} (not found in dumps)')
            continue

        d = load_dataset(dump_dir, ds, epoch=epoch)
        im = image_metrics(d['img_score'], d['img_label'])
        px = pixel_metrics(d['maps'], d['masks'])
        pro = aupro(d['maps'], d['masks'])
        pim = aupimo(d['maps'], d['masks'], d['img_label'])

        rows.append(dict(
            dataset=ds,
            img_auroc=round(100 * im['auroc'], 2),
            img_ap=round(100 * im['ap'], 2),
            px_auroc=round(100 * px['auroc'], 2),
            px_ap=round(100 * px['ap'], 2),
            aupro=round(100 * pro, 2) if pro == pro else 0,
            aupimo=round(100 * pim, 2) if pim == pim else 0,
        ))

    df = pd.DataFrame(rows)
    out = os.path.join(cfg['results_dir'], f'evaluation_{args.model}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f'\n{"=" * 60}')
    print(df.to_string(index=False))
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
