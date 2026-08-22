"""
data.py - Load and list dump datasets.

Standard dump format (one .npz per class):
    keys: masks (N,H,W), labels (N,), preds (N,H,W), preds_image (N,)
"""
import os
import glob
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

WRES = 256
PIX_SUBSAMPLE = 3_000_000


def _to3d(a):
    a = np.asarray(a)
    if a.ndim == 4:
        a = a[:, 0]
    return a.astype(np.float32)


def _resize(maps, out=WRES):
    if maps.shape[-1] == out and maps.shape[-2] == out:
        return maps
    if _HAS_TORCH:
        t = torch.from_numpy(maps).unsqueeze(1)
        t = torch.nn.functional.interpolate(
            t, size=(out, out), mode='bilinear', align_corners=False)
        return t.squeeze(1).numpy()
    ih = np.linspace(0, maps.shape[1] - 1, out).astype(int)
    iw = np.linspace(0, maps.shape[2] - 1, out).astype(int)
    return maps[:, ih][:, :, iw]


def list_datasets(dump_dir):
    return sorted([
        d for d in os.listdir(dump_dir)
        if os.path.isdir(os.path.join(dump_dir, d))
    ])


def _pick_epoch(files):
    eps = []
    for f in files:
        b = os.path.basename(f)
        if '_epoch' in b:
            try:
                eps.append(int(b.split('_epoch')[-1].split('.npz')[0]))
            except Exception:
                pass
    return max(eps) if eps else None


def load_dataset(dump_dir, dataset, epoch=None, wres=WRES, verbose=True):
    """Load all .npz dumps for a dataset, return merged arrays."""
    d = os.path.join(dump_dir, dataset)
    files = sorted(glob.glob(os.path.join(d, '*.npz')))
    if not files:
        raise FileNotFoundError('no npz in ' + d)

    if epoch is None:
        epoch = _pick_epoch(files)
    if epoch is not None:
        sel = [f for f in files if f.endswith('_epoch%d.npz' % epoch)]
        files = sel if sel else files

    M, GM, ISC, ILB, CID, names = [], [], [], [], [], []
    for ci, f in enumerate(files):
        b = os.path.basename(f)
        cname = b.split('_epoch')[0] if '_epoch' in b else b.replace('.npz', '')
        z = np.load(f)
        pm = _resize(_to3d(z['preds']), wres)
        gm = (_resize(_to3d(z['masks']), wres) > 0.5).astype(np.uint8)
        isc = np.asarray(z['preds_image']).astype(np.float32).reshape(-1)
        ilb = np.asarray(z['labels']).astype(np.int64).reshape(-1)
        n = min(len(isc), len(ilb), pm.shape[0], gm.shape[0])
        M.append(pm[:n])
        GM.append(gm[:n])
        ISC.append(isc[:n])
        ILB.append(ilb[:n])
        CID.append(np.full(n, ci, np.int64))
        names.append(cname)

    out = dict(
        maps=np.concatenate(M, 0),
        masks=np.concatenate(GM, 0),
        img_score=np.concatenate(ISC, 0),
        img_label=np.concatenate(ILB, 0),
        class_id=np.concatenate(CID, 0),
        class_names=names,
        epoch=epoch,
    )
    if verbose:
        print('%-16s imgs=%5d classes=%2d pos=%5d' % (
            dataset, len(out['img_label']), len(names),
            int(out['img_label'].sum())))
    return out


def load_classes(dump_dir, dataset, epoch=None, wres=WRES):
    """Return a list of per-class dicts WITHOUT pooling across classes."""
    d = os.path.join(dump_dir, dataset)
    files = sorted(glob.glob(os.path.join(d, '*.npz')))
    if not files:
        raise FileNotFoundError('no npz in ' + d)
    if epoch is None:
        epoch = _pick_epoch(files)
    if epoch is not None:
        sel = [f for f in files if f.endswith('_epoch%d.npz' % epoch)]
        files = sel if sel else files
    out = []
    for f in files:
        b = os.path.basename(f)
        cname = b.split('_epoch')[0] if '_epoch' in b else b.replace('.npz', '')
        z = np.load(f)
        pm = _resize(_to3d(z['preds']), wres)
        gm = (_resize(_to3d(z['masks']), wres) > 0.5).astype(np.uint8)
        isc = np.asarray(z['preds_image']).astype(np.float32).reshape(-1)
        ilb = np.asarray(z['labels']).astype(np.int64).reshape(-1)
        n = min(len(isc), len(ilb), pm.shape[0], gm.shape[0])
        out.append(dict(class_name=cname, maps=pm[:n], masks=gm[:n],
                        img_score=isc[:n], img_label=ilb[:n]))
    return out


def flat_subsample(maps, masks, cap=PIX_SUBSAMPLE, seed=0):
    y = masks.reshape(-1).astype(np.uint8)
    s = maps.reshape(-1).astype(np.float32)
    if len(y) > cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), cap, replace=False)
        y, s = y[idx], s[idx]
    return s, y
