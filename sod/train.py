
import os
import time
import csv
import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader import build_dataloaders
from sod_model   import SODNet, SODLoss
from evaluate    import compute_metrics   # reused here for live stats



# CHECKPOINT UTILITIES  (Bonus)

def save_checkpoint(state: dict, path: str):
    torch.save(state, path)
    print(f"  [Checkpoint] Saved → {path}")


def load_checkpoint(path: str, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch  = ckpt["epoch"] + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    print(f"  [Checkpoint] Resumed from epoch {ckpt['epoch']}  "
          f"(best_val_loss={best_val_loss:.4f})")
    return start_epoch, best_val_loss



# ONE EPOCH  (train or val)

def run_epoch(model, loader, loss_fn, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    all_iou, all_f1 = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, masks in loader:
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            preds = model(imgs)
            loss, _ = loss_fn(preds, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()

            # Compute batch-level metrics
            m = compute_metrics(preds.detach(), masks.detach())
            all_iou.append(m["iou"])
            all_f1.append(m["f1"])

    n   = len(loader)
    return {
        "loss" : total_loss / n,
        "iou"  : sum(all_iou) / len(all_iou),
        "f1"   : sum(all_f1)  / len(all_f1),
    }



# MAIN TRAINING FUNCTION

def train(
    image_dir:    str,
    mask_dir:     str,
    output_dir:   str  = "outputs",
    image_size:   int  = 224,
    batch_size:   int  = 16,
    num_workers:  int  = 2,
    base_filters: int  = 32,
    dropout_p:    float = 0.1,
    lr:           float = 1e-3,
    epochs:       int   = 20,
    patience:     int   = 5,
    resume:       bool  = False,
    seed:         int   = 42,
):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  SOD Training")
    print(f"  device={device}  image_size={image_size}  epochs={epochs}")
    print(f"{'='*60}\n")


    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path      = out / "last_checkpoint.pth"
    best_model_path = out / "best_model.pth"
    log_csv        = out / "training_log.csv"


    train_loader, val_loader, _ = build_dataloaders(
        image_dir, mask_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )


    model     = SODNet(base_filters=base_filters, dropout_p=dropout_p).to(device)
    loss_fn   = SODLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                              patience=2)

    print(f"  Model parameters: {model.count_parameters():,}\n")

    start_epoch   = 0
    best_val_loss = float("inf")
    no_improve    = 0

    if resume and ckpt_path.exists():
        start_epoch, best_val_loss = load_checkpoint(
            str(ckpt_path), model, optimizer, scheduler)
    else:
        # Write CSV header
        with open(log_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_iou", "train_f1",
                             "val_loss",   "val_iou",   "val_f1",
                             "lr", "time_s"])

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_m = run_epoch(model, train_loader, loss_fn, optimizer,
                            device, train=True)
        val_m   = run_epoch(model, val_loader,   loss_fn, optimizer,
                            device, train=False)

        scheduler.step(val_m["loss"])
        elapsed = time.time() - t0

        # Console log
        print(
            f"Epoch [{epoch+1:>3}/{epochs}]  "
            f"Train loss={train_m['loss']:.4f}  iou={train_m['iou']:.3f}  f1={train_m['f1']:.3f}  |  "
            f"Val   loss={val_m['loss']:.4f}  iou={val_m['iou']:.3f}  f1={val_m['f1']:.3f}  "
            f"lr={current_lr:.2e}  ({elapsed:.1f}s)"
        )

        # CSV log
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                f"{train_m['loss']:.6f}", f"{train_m['iou']:.6f}", f"{train_m['f1']:.6f}",
                f"{val_m['loss']:.6f}",   f"{val_m['iou']:.6f}",   f"{val_m['f1']:.6f}",
                f"{current_lr:.2e}", f"{elapsed:.1f}",
            ])

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            no_improve    = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✔ New best model saved (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1

        save_checkpoint({
            "epoch"          : epoch,
            "model_state"    : model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss"  : best_val_loss,
        }, str(ckpt_path))

        if no_improve >= patience:
            print(f"\n  Early stopping after {patience} epochs without improvement.")
            break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model  → {best_model_path}")
    print(f"Training log→ {log_csv}")
    return str(best_model_path)



# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SOD model")
    parser.add_argument("--image_dir",    required=True)
    parser.add_argument("--mask_dir",     required=True)
    parser.add_argument("--output_dir",   default="outputs")
    parser.add_argument("--image_size",   type=int,   default=224)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--num_workers",  type=int,   default=2)
    parser.add_argument("--base_filters", type=int,   default=32)
    parser.add_argument("--dropout_p",    type=float, default=0.1)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--patience",     type=int,   default=5)
    parser.add_argument("--resume",       action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    train(
        image_dir    = args.image_dir,
        mask_dir     = args.mask_dir,
        output_dir   = args.output_dir,
        image_size   = args.image_size,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        base_filters = args.base_filters,
        dropout_p    = args.dropout_p,
        lr           = args.lr,
        epochs       = args.epochs,
        patience     = args.patience,
        resume       = args.resume,
    )
