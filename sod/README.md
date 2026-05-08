# Salient Object Detection (SOD) — End-to-End Deep Learning Project

## Project Structure

```
sod_project/
├── data_loader.py        # Dataset loading, preprocessing & augmentation
├── sod_model.py          # CNN encoder-decoder model + loss function
├── train.py              # Training loop with checkpointing & early stopping
├── evaluate.py           # Metrics (IoU, F1, Precision, Recall, MAE) + visualisations
├── app.py                # Gradio demo app
├── demo_notebook.ipynb   # Jupyter demo notebook
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Step-by-Step Implementation Guide

### Step 1 — Set Up Your Environment

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

For Google Colab, add this at the top of your notebook:
```python
!pip install torch torchvision tqdm matplotlib pillow gradio -q
```

---

### Step 2 — Download a Dataset

Choose one of the following public datasets:

| Dataset   | Size   | Download Link |
|-----------|--------|---------------|
| ECSSD     | ~1,000 images | https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/ |
| DUTS-TR   | ~10,000 | http://saliencydetection.net/duts/ |
| MSRA10K   | ~10,000 | via Kaggle or academic mirrors |

**Expected folder structure after download:**
```
dataset/
  images/    ← RGB images  (.jpg)
  masks/     ← Binary masks (.png, same filename stem)
```

**Quick test with ECSSD (recommended for beginners — small but clean):**
```bash
# After downloading and unzipping ECSSD:
# images → dataset/images/
# masks  → dataset/masks/
```

---

### Step 3 — Verify the Data Pipeline

```bash
python data_loader.py
```

This will:
- Count image/mask pairs
- Apply augmentations to a sample
- Save `sample_check.png` so you can visually verify
- Print batch shapes

---

### Step 4 — Verify the Model

```bash
python sod_model.py
```

Expected output:
```
Running on: cuda   (or cpu)
Output shape : torch.Size([4, 1, 224, 224])
Loss         : 0.8432  {'bce': 0.6931, 'iou_loss': 0.3001}
Params       : 1,937,441
```

---

### Step 5 — Train the Model

```bash
python train.py \
  --image_dir dataset/images \
  --mask_dir  dataset/masks  \
  --output_dir outputs        \
  --image_size 224            \
  --batch_size 16             \
  --epochs 20                 \
  --patience 5
```

**Key outputs saved to `outputs/`:**
- `best_model.pth`      — best validation loss checkpoint
- `last_checkpoint.pth` — latest checkpoint (for resume)
- `training_log.csv`    — per-epoch metrics

**Resume interrupted training (Bonus feature):**
```bash
python train.py --image_dir ... --mask_dir ... --resume
```

---

### Step 6 — Evaluate on the Test Set

```bash
python evaluate.py \
  --model_path  outputs/best_model.pth \
  --image_dir   dataset/images          \
  --mask_dir    dataset/masks           \
  --output_dir  outputs                 \
  --log_csv     outputs/training_log.csv
```

**Outputs:**
- `outputs/evaluation_results.txt` — numeric metrics
- `outputs/predictions.png`        — visual grid (input/GT/pred/overlay)
- `outputs/training_curves.png`    — loss & IoU curves

---

### Step 7 — Run the Demo

**Option A — Gradio web app:**
```bash
python app.py --model_path outputs/best_model.pth
# Opens in browser at http://localhost:7860
```

**Option B — Jupyter Notebook:**
```bash
jupyter notebook demo_notebook.ipynb
```
Run cells in order, then upload any image.

---

## Experiments & Improvements (Required)

Run at least two experiments and compare results:

### Experiment 1 — Add Dropout (already in model)
```bash
python train.py --dropout_p 0.2 --output_dir outputs/exp1_dropout
```

### Experiment 2 — Deeper network (more filters)
```bash
python train.py --base_filters 64 --output_dir outputs/exp2_deeper
```

### Experiment 3 — Lower learning rate
```bash
python train.py --lr 0.0003 --output_dir outputs/exp3_lr
```

Fill in your comparison table in the report:

| Config           | Val IoU | Val F1 | Test IoU | Test F1 |
|------------------|---------|--------|----------|---------|
| Baseline (f=32)  |         |        |          |         |
| + Dropout (p=0.2)|         |        |          |         |
| Deeper (f=64)    |         |        |          |         |

---

## Model Architecture Summary

```
Input [B, 3, 224, 224]
    ↓
Encoder Block 1: Conv→BN→ReLU→Conv→BN→ReLU→MaxPool  [→ 112×112, 32 ch]
Encoder Block 2:                                       [→  56× 56, 64 ch]
Encoder Block 3:                                       [→  28× 28, 128 ch]
Encoder Block 4:                                       [→  14× 14, 256 ch]
    ↓
Bottleneck:       Conv→BN→ReLU×2                      [    14× 14, 512 ch]
    ↓
Decoder Block 4:  ConvTranspose + skip + Conv×2        [→  28× 28, 256 ch]
Decoder Block 3:                                       [→  56× 56, 128 ch]
Decoder Block 2:                                       [→ 112×112,  64 ch]
Decoder Block 1:                                       [→ 224×224,  32 ch]
    ↓
Head:             Conv 1×1 → Sigmoid
    ↓
Output [B, 1, 224, 224]  — saliency probability mask
```

**Loss:** `BCE + 0.5 × (1 - IoU)`  
**Optimizer:** `Adam(lr=1e-3, weight_decay=1e-4)`  
**Scheduler:** `ReduceLROnPlateau(factor=0.5, patience=2)`

---

## Tips

- Start with **ECSSD** (~1,000 images) for fast iteration, then switch to DUTS for better results.
- If training on CPU, use `--image_size 128 --batch_size 8` to keep it manageable.
- Always run `data_loader.py` first to confirm your folder paths are correct.
- Intermediate visualisations (`sample_check.png`, `predictions.png`) are your best debugging tool.
- Back up `outputs/` daily or push to GitHub regularly.
