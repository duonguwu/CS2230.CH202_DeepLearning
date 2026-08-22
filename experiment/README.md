# CalZSAD Experiment - Reproduce AA-CLIP + Post-processing

Reproduce kết quả AA-CLIP trên 6 datasets (BTAD, MPDD, MVTec-AD, Brain, Liver, Retina),
sau đó áp dụng CalZSAD post-processing để chọn ngưỡng tự động.

## Yêu cầu

- Python >= 3.9
- GPU >= 16GB VRAM (RTX 3090/4090/5090)
- CLIP weight ViT-L-14-336px (~890MB, tự download lần đầu)

## Cài đặt

```bash
# 1. Clone AA-CLIP source code
git clone https://github.com/ByGary/AA-CLIP.git AA-CLIP
cd AA-CLIP
pip install -r requirements.txt
cd ..

# 2. Cài dependencies cho experiment
pip install -r experiment/requirements.txt
```

## Cấu trúc

```
experiment/
├── lib/                    # Thư viện dùng chung
│   ├── data.py             # Load dump .npz files
│   ├── metrics.py          # AUROC, AP, AUPRO, AUPIMO, F1, FPR
│   └── calzsad.py          # CalZSAD core: top-1% signal, robust norm, conformal threshold
├── scripts/
│   ├── run_inference.py    # Chạy AA-CLIP inference trên tất cả datasets
│   ├── evaluate.py         # Tính ranking metrics (AUROC, AP, AUPRO, AUPIMO)
│   └── calzsad_threshold.py # Chạy CalZSAD few-shot thresholding
├── configs/
│   └── paths.yaml          # Config đường dẫn (SỬA TRƯỚC KHI CHẠY)
├── results/                # Output CSV
├── figs/                   # Output hình
└── README.md               # File này
```

## Các bước chạy

### Bước 0: Cấu hình đường dẫn

Sửa file `configs/paths.yaml` cho đúng đường dẫn trên máy:

```yaml
data_root: /path/to/datasets          # thư mục chứa các dataset
aaclip_repo: /path/to/AA-CLIP         # thư mục AA-CLIP source code
aaclip_checkpoint: /path/to/checkpoints/aaclip  # chứa text_adapter.pth + image_adapter_20.pth
dump_root: ./dumps                    # output anomaly score maps
results_dir: ./results                # output CSV
figs_dir: ./figs                      # output hình
```

### Bước 1: Tải datasets

Xem file `DATASETS.md` ở thư mục gốc.

### Bước 2: Chạy AA-CLIP inference

```bash
# Inference trên tất cả 6 datasets
CUDA_VISIBLE_DEVICES=0 python scripts/run_inference.py

# Hoặc chạy từng dataset
CUDA_VISIBLE_DEVICES=0 python scripts/run_inference.py --datasets BTAD MPDD
```

Output: `dumps/aaclip/<dataset>/<class>_epoch20.npz` cho mỗi class.

### Bước 3: Tính ranking metrics

```bash
python scripts/evaluate.py --model aaclip
```

Output: `results/evaluation_aaclip.csv` chứa img-AUROC, img-AP, px-AUROC, px-AP, AUPRO, AUPIMO.

### Bước 4: Chạy CalZSAD thresholding

```bash
python scripts/calzsad_threshold.py --model aaclip --K 20 --seeds 8
```

Output: `results/calzsad_aaclip.csv` chứa F1, Recall, FPR, Oracle F1, Dice@FAR1%.

## Kết quả mong đợi

### Image-level (AA-CLIP, K=20, FPR target 5%)

| Dataset  | img-AUROC | img-AP | FPR(%) | Recall(%) | F1(%) | Oracle F1(%) |
|----------|-----------|--------|--------|-----------|-------|-------------|
| BTAD     | 93.98     | 93.09  | 5.74   | 64.47     | 74.26 | 79.69       |
| MPDD     | 73.06     | 78.49  | 3.79   | 30.27     | 46.21 | 49.62       |
| MVTec-AD | 84.92     | 92.94  | 4.41   | 72.74     | 83.70 | 86.47       |
| Brain    | 58.40     | 86.30  | 3.43   | 21.41     | 34.26 | 42.92       |
| Liver    | 57.16     | 53.74  | 0.94   | 2.16      | 4.09  | 13.55       |
| Retina   | 71.49     | 65.50  | 5.02   | 51.54     | 63.81 | 71.94       |

### Pixel-level (AA-CLIP)

| Dataset  | px-AUROC | px-AP  | AUPRO  | AUPIMO | Dice@FAR1% | px-FPR(%) |
|----------|----------|--------|--------|--------|------------|-----------|
| BTAD     | 97.41    | 55.66  | 67.38  | 75.95  | 24.30      | 1.01      |
| MPDD     | 96.55    | 25.49  | 81.38  | 42.25  | 22.15      | 0.87      |
| MVTec-AD | 92.55    | 48.97  | 74.29  | 78.12  | 35.12      | 0.89      |
| Brain    | 95.21    | 39.21  | 74.68  | 35.46  | 31.11      | 1.09      |
| Liver    | 97.71    | 9.96   | 64.39  | 47.48  | 23.54      | 0.96      |
| Retina   | 96.25    | 63.54  | 74.90  | 47.94  | 50.11      | 0.93      |
