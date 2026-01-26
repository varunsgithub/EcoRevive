# 🔥 California Fire Detection Model

## Overview

A focused, production-ready model for California wildfire detection and recovery monitoring.
This model learns from **continuous burn severity** labels (not binary 0/1) and is validated
on held-out fire events to ensure generalization.

---

## 🎯 What Makes This Different

| Previous Approach | This Approach |
|------------------|---------------|
| Binary labels (0/1) | **Continuous severity (0-100%)** via dNBR |
| 8 different ecosystems | **California-specific** vegetation types |
| Arbitrary normalization | **Computed from training data** |
| Single timestamp | **Temporal pairs** (before/after fire) |
| Overall validation | **Per-fire validation** with held-out events |
| Complex multi-loss | **Simple BCE → add complexity only if needed** |

---

## 📁 Folder Structure

```
California-Fire-Model/
├── README.md                    # This file
├── config.py                    # All configuration in one place
│
├── data/
│   ├── download_fire_data.py   # Collect Sentinel-2 + burn severity
│   ├── compute_statistics.py   # Calculate actual band means/stds
│   ├── validate_dataset.py     # Quality check before training
│   └── dataset.py              # PyTorch dataset with proper loading
│
├── model/
│   ├── architecture.py         # Clean U-Net implementation
│   ├── losses.py               # Simple, focused loss functions
│   └── metrics.py              # IoU, MAE, per-class metrics
│
├── training/
│   ├── train.py                # Main training loop
│   ├── validate.py             # Validation with per-fire breakdown
│   └── callbacks.py            # Early stopping, checkpointing
│
├── inference/
│   ├── predict.py              # Run inference on new images
│   └── visualize.py            # Generate prediction visualizations
│
└── notebooks/
    ├── 01_explore_data.ipynb   # Understand the dataset
    ├── 02_train_model.ipynb    # Training notebook for Colab
    └── 03_demo.ipynb           # Hackathon demo notebook
```

---

## 🌲 California Focus Areas

### Major Fire Events (Training Data)

| Fire Name | Year | Acres | County | Notes |
|-----------|------|-------|--------|-------|
| Dixie Fire | 2021 | 963,309 | Butte, Plumas, Lassen, Shasta, Tehama | 2nd largest in CA history |
| Caldor Fire | 2021 | 221,835 | El Dorado, Amador, Alpine | Crossed Sierra Nevada |
| Creek Fire | 2020 | 379,895 | Fresno, Madera | Created fire tornado |
| Camp Fire | 2018 | 153,336 | Butte | Destroyed Paradise |
| Mendocino Complex | 2018 | 459,123 | Mendocino, Lake, Colusa, Glenn | Largest in CA history |
| Thomas Fire | 2017 | 281,893 | Ventura, Santa Barbara | Largest at the time |
| Rim Fire | 2013 | 257,314 | Tuolumne | Near Yosemite |

### Held-Out for Testing (NOT used in training)
- **Kincade Fire** (2019) - Sonoma County
- **Woolsey Fire** (2018) - Los Angeles/Ventura

### Vegetation Types
- Conifer Forest (Sierra Nevada)
- Oak Woodland (Foothills)
- Chaparral (Southern CA)
- Grassland (Central Valley edges)

---

## 📊 Label Strategy: Continuous Burn Severity

Instead of binary labels, we use **dNBR (differenced Normalized Burn Ratio)**:

```
NBR = (NIR - SWIR) / (NIR + SWIR)
    = (B8 - B12) / (B8 + B12)

Pre-fire NBR → Post-fire NBR → dNBR = Pre - Post

dNBR Ranges:
  0.00 - 0.10 : Unburned
  0.10 - 0.27 : Low severity
  0.27 - 0.44 : Moderate-low severity
  0.44 - 0.66 : Moderate-high severity
  0.66 - 1.00+: High severity
```

This gives us a **continuous 0-1 regression target** instead of binary classification.

---

## 🔄 Data Collection Strategy

### For Each Fire:
1. **Pre-fire image** (1-2 months before) - baseline vegetation
2. **Post-fire image** (1-2 months after) - fresh burn scar
3. **Recovery images** (optional) - 1yr, 2yr, 3yr later

### Sources:
- **Sentinel-2 L2A** - 10m resolution, 10 spectral bands
- **MTBS (Monitoring Trends in Burn Severity)** - official burn perimeters
- **RAVG (Rapid Assessment of Vegetation)** - detailed severity maps

---

## 🚀 Quick Start

### Step 1: Download Data
```bash
cd California-Fire-Model/data
python download_fire_data.py
```

### Step 2: Compute Normalization Statistics
```bash
python compute_statistics.py
```

### Step 3: Validate Dataset
```bash
python validate_dataset.py
```

### Step 4: Train Model
```bash
cd ../training
python train.py --epochs 50 --batch-size 16
```

### Step 5: Run Inference
```bash
cd ../inference
python predict.py --checkpoint ../checkpoints/best.pth --input path/to/image.tif
```

---

## ⚙️ Configuration

All settings are in `config.py`:

```python
# Data
FIRES = ['dixie', 'caldor', 'creek', 'camp', ...]  # Training fires
TEST_FIRES = ['kincade', 'woolsey']                 # Held-out fires

# Model
INPUT_CHANNELS = 10      # Sentinel-2 bands
BASE_CHANNELS = 64       # U-Net width
USE_ATTENTION = True     # Attention gates

# Training
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Computed (don't edit - run compute_statistics.py)
BAND_MEANS = [...]       # Calculated from training data
BAND_STDS = [...]        # Calculated from training data
```

---

## 📈 Expected Results

With this approach, you should achieve:

| Metric | Target | Notes |
|--------|--------|-------|
| MAE (severity) | < 0.10 | Mean absolute error on 0-1 scale |
| IoU (>50% burn) | > 0.75 | Intersection over union for burned areas |
| Per-fire variance | < 0.05 | Consistent across different fires |
| Held-out fire IoU | > 0.70 | Generalization to unseen fires |

---

## 🏆 Hackathon Demo Strategy

1. **Show temporal progression:**
   - Healthy forest → Fire → 1 year recovery → 2 year recovery

2. **Interactive map:**
   - User clicks location → model predicts severity → show confidence

3. **Before/after slider:**
   - Dramatic visual of prediction accuracy

4. **Recovery tracking:**
   - "This area has recovered 65% since the 2021 fire"

---

## 📚 References

- [MTBS Data Access](https://www.mtbs.gov/direct-download)
- [Sentinel-2 Band Info](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spectral)
- [dNBR Interpretation](https://www.earthdatascience.org/courses/use-data-open-source-python/multispectral-remote-sensing/vegetation-indices-in-python/calculate-dNBR-Landsat-8/)
