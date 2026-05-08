

import os
import random
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.transforms.functional as TF



# 1. PAIRED AUGMENTATION  

class PairedAugmentation:

    def __init__(self, image_size: int = 224, augment: bool = True):
        self.image_size = image_size
        self.augment = augment

    def __call__(self, image: Image.Image, mask: Image.Image):
        # --- Resize ---
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask  = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        if self.augment:
            # Horizontal flip (50 % chance)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            # Random crop → resize back
            if random.random() > 0.5:
                i, j, h, w = T.RandomCrop.get_params(
                    image, output_size=(int(self.image_size * 0.85),
                                        int(self.image_size * 0.85))
                )
                image = TF.resized_crop(image, i, j, h, w,
                                        (self.image_size, self.image_size),
                                        Image.BILINEAR)
                mask  = TF.resized_crop(mask,  i, j, h, w,
                                        (self.image_size, self.image_size),
                                        Image.NEAREST)

            # Random rotation ±15 °
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
                mask  = TF.rotate(mask,  angle, interpolation=Image.NEAREST)

            # Brightness / contrast / saturation – image only
            image = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)(image)

        # --- To tensor + normalise ---
        image_tensor = TF.to_tensor(image)                    # [3, H, W], range [0,1]
        image_tensor = TF.normalize(image_tensor,
                                    mean=[0.485, 0.456, 0.406],
                                    std =[0.229, 0.224, 0.225])

        mask_array  = np.array(mask, dtype=np.float32) / 255.0  # [H, W]
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)  # [1, H, W]
        mask_tensor = (mask_tensor > 0.5).float()                # binarise

        return image_tensor, mask_tensor



# 2. DATASET CLASS

class SODDataset(Dataset):
    """
    dataset class
    Supported datasets: DUTS, ECSSD, MSRA10K
    """

    def __init__(self,
                 image_dir: str,
                 mask_dir: str,
                 image_size: int = 224,
                 augment: bool = True):

        self.image_dir  = Path(image_dir)
        self.mask_dir   = Path(mask_dir)
        self.transform  = PairedAugmentation(image_size, augment)

        # Collect matching pairs
        img_exts  = {".jpg", ".jpeg", ".png", ".bmp"}
        self.pairs = []

        for img_path in sorted(self.image_dir.iterdir()):
            if img_path.suffix.lower() not in img_exts:
                continue
            # Try matching mask by same stem (any extension)
            mask_path = self._find_mask(img_path.stem)
            if mask_path is not None:
                self.pairs.append((img_path, mask_path))

        if len(self.pairs) == 0:
            raise FileNotFoundError(
                f"No image/mask pairs found in:\n"
                f"  images → {self.image_dir}\n"
                f"  masks  → {self.mask_dir}"
            )
        print(f"[SODDataset] Found {len(self.pairs)} pairs "
              f"(augment={augment}, size={image_size})")

    def _find_mask(self, stem: str) -> Path | None:
        for ext in [".png", ".jpg", ".bmp"]:
            candidate = self.mask_dir / (stem + ext)
            if candidate.exists():
                return candidate
        return None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")   # grey-scale

        image_tensor, mask_tensor = self.transform(image, mask)
        return image_tensor, mask_tensor



# 3. SPLIT & DATALOADER FACTORY

def build_dataloaders(
    image_dir: str,
    mask_dir: str,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 2,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed: int = 42,
):
    
    torch.manual_seed(seed)

    # Full dataset with NO augmentation first (to get total size)
    full_dataset = SODDataset(image_dir, mask_dir,
                              image_size=image_size, augment=False)
    n = len(full_dataset)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    n_test  = n - n_train - n_val

    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    # Augmented train dataset (re-use same pairs, different transform)
    train_dataset = SODDataset(image_dir, mask_dir,
                               image_size=image_size, augment=True)
    val_dataset   = SODDataset(image_dir, mask_dir,
                               image_size=image_size, augment=False)
    test_dataset  = SODDataset(image_dir, mask_dir,
                               image_size=image_size, augment=False)

    from torch.utils.data import Subset
    train_loader = DataLoader(Subset(train_dataset, train_idx),
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(Subset(val_dataset, val_idx),
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(Subset(test_dataset, test_idx),
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"\n[DataLoaders] train={n_train}  val={n_val}  test={n_test}")
    return train_loader, val_loader, test_loader



# 4. QUICK SANITY CHECK

def visualise_sample(image_dir: str, mask_dir: str, image_size: int = 224):

    import matplotlib.pyplot as plt

    ds = SODDataset(image_dir, mask_dir, image_size=image_size, augment=True)
    img, mask = ds[0]

    # De-normalise image for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_display = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_display);  axes[0].set_title("Image");       axes[0].axis("off")
    axes[1].imshow(mask.squeeze().numpy(), cmap="gray")
    axes[1].set_title("Saliency Mask"); axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("sample_check.png", dpi=120)
    plt.show()
    print("Saved sample_check.png")


if __name__ == "__main__":
    IMAGE_DIR = r"C:\Users\adeli\Downloads\sod\dataset\images"
    MASK_DIR  = r"C:\Users\adeli\Downloads\sod\dataset\masks"

    visualise_sample(IMAGE_DIR, MASK_DIR)
    train_loader, val_loader, test_loader = build_dataloaders(
        IMAGE_DIR, MASK_DIR, image_size=224, batch_size=16
    )
    imgs, masks = next(iter(train_loader))
    print(f"Batch shapes — images: {imgs.shape}  masks: {masks.shape}")
