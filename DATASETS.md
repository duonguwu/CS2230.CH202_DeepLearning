# Datasets

Dự án sử dụng 6 bộ dữ liệu: 3 industrial + 3 medical.

## Download từ Kaggle

```bash
# Cài Kaggle CLI (1 lần)
pip install kaggle
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 1. MVTec-AD (15 classes, Industrial)
kaggle datasets download ipythonx/mvtec-ad -p datasets/ --unzip

# 2. BTAD (3 classes, Industrial)
kaggle datasets download duongkaiba/btad-dataset -p datasets/ --unzip

# 3. MPDD (6 classes, Industrial)
kaggle datasets download duongkaiba/mpdd-dataset -p datasets/ --unzip

# 4. Brain MRI, Liver CT, Retina OCT (Medical - BMAD benchmark)
kaggle datasets download duongkaiba/bmad-aaclip -p datasets/ --unzip

# 5. VisA (dùng cho train AA-CLIP, không nằm trong 6 bộ báo cáo)
kaggle datasets download tensura3607/amazon-visa-anomaly -p datasets/ --unzip
```

## Cấu trúc sau khi tải

```
datasets/
├── MVTec/
│   ├── bottle/
│   ├── cable/
│   ├── capsule/
│   └── ... (15 classes)
├── BTAD/
│   ├── 01/
│   ├── 02/
│   └── 03/
├── MPDD/
│   ├── bracket_black/
│   ├── bracket_brown/
│   ├── bracket_white/
│   ├── connector/
│   ├── metal_plate/
│   └── tubes/
├── Brain_AD/
│   ├── test/
│   └── train/
├── Liver_AD/
│   ├── test/
│   └── train/
├── Retina_RESC_AD/
│   ├── test/
│   └── train/
└── VisA/
    ├── candle/
    ├── capsules/
    └── ... (12 classes)
```

Mỗi class thường chứa:
- `train/good/` hoặc `train/normal/` : ảnh bình thường
- `test/good/` + `test/<defect_type>/` : ảnh test (normal + anomaly)
- `ground_truth/` : mask pixel-level (nếu có)
