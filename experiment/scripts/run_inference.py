#!/usr/bin/env python3
"""
run_inference.py - Run AA-CLIP zero-shot inference on target datasets.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_inference.py
    CUDA_VISIBLE_DEVICES=0 python scripts/run_inference.py --datasets BTAD MPDD
"""
import os
import sys
import argparse
import glob
import shutil
import subprocess
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load config
_cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'paths.yaml')
with open(_cfg_path) as f:
    cfg = yaml.safe_load(f)

AACLIP_DIR = cfg['aaclip_repo']
CKPT_DIR = cfg['aaclip_checkpoint']
DUMP_OUT = os.path.join(cfg['dump_root'], 'aaclip')
DATA_ROOT = cfg['data_root']

# Only 6 datasets for the report + VisA for training
DATASETS = ['VisA', 'MVTec', 'BTAD', 'MPDD', 'Brain', 'Liver', 'Retina']

# AA-CLIP inference hyperparameters (from original paper)
IMG_SIZE = 518
BATCH = 16
SEED = 111
TEXT_ADAPT_UNTIL = 3
IMAGE_ADAPT_UNTIL = 6
TEXT_ADAPT_WEIGHT = 0.1
IMAGE_ADAPT_WEIGHT = 0.1


def check_prerequisites():
    """Verify checkpoint files exist."""
    text_ckpt = os.path.join(CKPT_DIR, 'text_adapter.pth')
    if not os.path.exists(text_ckpt):
        raise FileNotFoundError(f'text_adapter.pth not found in {CKPT_DIR}')

    img_ckpts = glob.glob(os.path.join(CKPT_DIR, 'image_adapter_*.pth'))
    if not img_ckpts:
        plain = os.path.join(CKPT_DIR, 'image_adapter.pth')
        if os.path.exists(plain):
            target = os.path.join(CKPT_DIR, 'image_adapter_20.pth')
            shutil.copy(plain, target)
            print(f'Created {os.path.basename(target)} from image_adapter.pth')
        else:
            raise FileNotFoundError(f'No image_adapter checkpoint in {CKPT_DIR}')

    print(f'Checkpoint: {CKPT_DIR}')
    print(f'AA-CLIP repo: {AACLIP_DIR}')
    print(f'Data root: {DATA_ROOT}')


def write_aaclip_config():
    """Write config.yaml for AA-CLIP with correct dataset paths."""
    config = {
        'paths': {
            'base_path': DATA_ROOT,
            'datasets': cfg['dataset_paths'],
        }
    }
    config_path = os.path.join(AACLIP_DIR, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, allow_unicode=True)
    print(f'Wrote {config_path}')


def run_inference(datasets=None):
    """Run test.py for each dataset."""
    targets = datasets or DATASETS
    for ds in targets:
        print(f'\n{"=" * 50}')
        print(f'INFERENCE: {ds}')
        print(f'{"=" * 50}')

        cmd = [
            sys.executable, 'test.py',
            '--dataset', ds,
            '--save_path', CKPT_DIR,
            '--img_size', str(IMG_SIZE),
            '--text_adapt_until', str(TEXT_ADAPT_UNTIL),
            '--image_adapt_until', str(IMAGE_ADAPT_UNTIL),
            '--text_adapt_weight', str(TEXT_ADAPT_WEIGHT),
            '--image_adapt_weight', str(IMAGE_ADAPT_WEIGHT),
            '--batch_size', str(BATCH),
            '--seed', str(SEED),
        ]
        result = subprocess.run(cmd, cwd=AACLIP_DIR)
        if result.returncode != 0:
            print(f'{ds} failed with exit code {result.returncode}')
        else:
            print(f'{ds} done')


def collect_dumps():
    """Copy pred_cache from checkpoint dir to dumps/aaclip."""
    src = os.path.join(CKPT_DIR, 'pred_cache')
    if not os.path.exists(src):
        print(f'pred_cache not found at {src}')
        return
    if os.path.exists(DUMP_OUT):
        shutil.rmtree(DUMP_OUT)
    shutil.copytree(src, DUMP_OUT)
    print(f'Dumps copied to {DUMP_OUT}')

    for ds in sorted(os.listdir(DUMP_OUT)):
        ds_path = os.path.join(DUMP_OUT, ds)
        if os.path.isdir(ds_path):
            npz_files = [f for f in os.listdir(ds_path) if f.endswith('.npz')]
            print(f'   {ds}: {len(npz_files)} classes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AA-CLIP inference')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to run (default: all)')
    args = parser.parse_args()

    check_prerequisites()
    write_aaclip_config()
    run_inference(datasets=args.datasets)
    collect_dumps()
    print('\nAll done!')
