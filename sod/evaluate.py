

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from data_loader import build_dataloaders
from sod_model   import SODNet



# METRIC FUNCTIONS  

def compute_metrics(pred: torch.Tensor,
                    target: torch.Tensor,
                    threshold: float = 0.5,
                    smooth: float = 1e-6) -> dict:
   
    pred_bin = (pred > threshold).float()

    pred_flat   = pred_bin.view(pred_bin.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    tp    = (pred_flat * target_flat).sum(dim=1)
    fp    = (pred_flat * (1 - target_flat)).sum(dim=1)
    fn    = ((1 - pred_flat) * target_flat).sum(dim=1)

    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn  + smooth)
    f1        = 2 * precision * recall / (precision + recall + smooth)
    iou       = (tp + smooth) / (tp + fp + fn + smooth)

    mae_batch = (pred - target).abs().mean(dim=[1, 2, 3])

    return {
        "iou"      : iou.mean().item(),
        "precision": precision.mean().item(),
        "recall"   : recall.mean().item(),
        "f1"       : f1.mean().item(),
        "mae"      : mae_batch.mean().item(),
    }



# FULL TEST-SET EVALUATION

def evaluate(model_path: str,
             image_dir: str,
             mask_dir: str,
             output_dir: str = "outputs",
             image_size: int = 224,
             batch_size: int = 16,
             num_workers: int = 2,
             base_filters: int = 32,
             dropout_p: float = 0.0):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out    = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)


    _, _, test_loader = build_dataloaders(
        image_dir, mask_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = SODNet(base_filters=base_filters, dropout_p=dropout_p).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    
    all_metrics = {"iou": [], "precision": [], "recall": [], "f1": [], "mae": []}
    sample_imgs, sample_masks, sample_preds = [], [], []

    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="Evaluating"):
            imgs  = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)

            m = compute_metrics(preds, masks)
            for k, v in m.items():
                all_metrics[k].append(v)

            # Collect first batch for visualisation
            if len(sample_imgs) == 0:
                sample_imgs  = imgs.cpu()
                sample_masks = masks.cpu()
                sample_preds = preds.cpu()


    print("\n" + "="*50)
    print("  TEST SET RESULTS")
    print("="*50)
    results = {}
    for k, vals in all_metrics.items():
        results[k] = np.mean(vals)
        print(f"  {k.upper():12s}: {results[k]:.4f}")
    print("="*50 + "\n")

    # Save to txt
    with open(out / "evaluation_results.txt", "w") as f:
        f.write("TEST SET EVALUATION RESULTS\n")
        f.write("="*40 + "\n")
        for k, v in results.items():
            f.write(f"{k.upper():12s}: {v:.4f}\n")

    visualise_predictions(
        sample_imgs, sample_masks, sample_preds,
        save_path=str(out / "predictions.png"),
        n_show=min(4, sample_imgs.size(0)),
    )

    return results



# VISUALISATION

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denorm(img_tensor: torch.Tensor) -> np.ndarray:
    """De-normalise a [3,H,W] tensor → [H,W,3] uint8 numpy array."""
    img = (img_tensor * _STD + _MEAN).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def visualise_predictions(imgs:  torch.Tensor,
                          masks: torch.Tensor,
                          preds: torch.Tensor,
                          save_path: str = "predictions.png",
                          n_show: int = 4,
                          threshold: float = 0.5):
   
    n = min(n_show, imgs.size(0))
    cols = ["Input Image", "Ground Truth", "Prediction", "Overlay"]

    fig = plt.figure(figsize=(4 * len(cols), 4 * n))
    gs  = gridspec.GridSpec(n, len(cols), hspace=0.05, wspace=0.05)

    for row in range(n):
        img_np   = denorm(imgs[row])
        mask_np  = masks[row].squeeze().numpy()
        pred_np  = preds[row].squeeze().numpy()
        pred_bin = (pred_np > threshold).astype(np.float32)

        # Overlay: green channel highlight
        overlay  = img_np.copy().astype(np.float32)
        overlay[..., 1] = np.clip(overlay[..., 1] + pred_bin * 120, 0, 255)
        overlay = overlay.astype(np.uint8)

        for col, (data, cmap) in enumerate([
            (img_np,  None),
            (mask_np, "gray"),
            (pred_np, "gray"),
            (overlay, None),
        ]):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1 if cmap else None)
            ax.axis("off")
            if row == 0:
                ax.set_title(cols[col], fontsize=11, pad=4)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved visualisation → {save_path}")


def plot_training_curves(log_csv: str, save_path: str = "training_curves.png"):

    import csv

    epochs, train_loss, val_loss = [], [], []
    train_iou, val_iou           = [], []

    with open(log_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_iou.append(float(row["train_iou"]))
            val_iou.append(float(row["val_iou"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, label="Train Loss")
    ax1.plot(epochs, val_loss,   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves"); ax1.legend()

    ax2.plot(epochs, train_iou, label="Train IoU")
    ax2.plot(epochs, val_iou,   label="Val IoU")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("IoU")
    ax2.set_title("IoU Curves"); ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved training curves → {save_path}")



# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SOD model")
    parser.add_argument("--model_path",   required=True)
    parser.add_argument("--image_dir",    required=True)
    parser.add_argument("--mask_dir",     required=True)
    parser.add_argument("--output_dir",   default="outputs")
    parser.add_argument("--image_size",   type=int,   default=224)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--num_workers",  type=int,   default=2)
    parser.add_argument("--base_filters", type=int,   default=32)
    parser.add_argument("--log_csv",      default=None,
                        help="Path to training_log.csv to plot curves")
    args = parser.parse_args()

    evaluate(
        model_path   = args.model_path,
        image_dir    = args.image_dir,
        mask_dir     = args.mask_dir,
        output_dir   = args.output_dir,
        image_size   = args.image_size,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        base_filters = args.base_filters,
    )

    if args.log_csv:
        plot_training_curves(args.log_csv,
                             save_path=str(Path(args.output_dir) / "training_curves.png"))
