
import torch
import torch.nn as nn
import torch.nn.functional as F


# BUILDING BLOCKS

class ConvBNReLU(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch,  out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)
        self.drop  = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.pool  = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        return self.pool(x), x   # (down-sampled, skip-connection)


class DecoderBlock(nn.Module):

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 dropout_p: float = 0.0):
        super().__init__()
        self.up    = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv1 = ConvBNReLU(out_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)
        self.drop  = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x, skip):
        x = self.up(x)
        # Handle odd spatial sizes from encoder
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x



# FULL MODEL

class SODNet(nn.Module):
    
    def __init__(self,
                 in_channels: int  = 3,
                 base_filters: int = 32,
                 dropout_p: float  = 0.0):
        super().__init__()

        f = base_filters   # shorthand

        self.enc1 = EncoderBlock(in_channels, f,      dropout_p)   # 224→112
        self.enc2 = EncoderBlock(f,           f * 2,  dropout_p)   # 112→56
        self.enc3 = EncoderBlock(f * 2,       f * 4,  dropout_p)   # 56→28
        self.enc4 = EncoderBlock(f * 4,       f * 8,  dropout_p)   # 28→14

        
        self.bottleneck = nn.Sequential(
            ConvBNReLU(f * 8,  f * 16),
            ConvBNReLU(f * 16, f * 16),
            nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity(),
        )

        self.dec4 = DecoderBlock(f * 16, f * 8,  f * 8,  dropout_p)  # 14→28
        self.dec3 = DecoderBlock(f * 8,  f * 4,  f * 4,  dropout_p)  # 28→56
        self.dec2 = DecoderBlock(f * 4,  f * 2,  f * 2,  dropout_p)  # 56→112
        self.dec1 = DecoderBlock(f * 2,  f,      f,       dropout_p)  # 112→224

        self.head = nn.Sequential(
            nn.Conv2d(f, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Encoder (returns pool output + skip)
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)

        # Bottleneck
        b = self.bottleneck(x4)

        # Decoder
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.head(d1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



# LOSS FUNCTION

class SODLoss(nn.Module):
  

    def __init__(self, bce_weight: float = 1.0, iou_weight: float = 0.5,
                 smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.smooth     = smooth
        self.bce        = nn.BCELoss()

    def iou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        inter  = (pred * target).sum(dim=1)
        union  = pred.sum(dim=1) + target.sum(dim=1) - inter
        iou    = (inter + self.smooth) / (union + self.smooth)
        return 1.0 - iou.mean()

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        bce  = self.bce(pred, target)
        iou  = self.iou_loss(pred, target)
        loss = self.bce_weight * bce + self.iou_weight * iou
        return loss, {"bce": bce.item(), "iou_loss": iou.item()}



# QUICK TEST

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    model      = SODNet(base_filters=32, dropout_p=0.1).to(device)
    loss_fn    = SODLoss()
    dummy_img  = torch.randn(4, 3, 224, 224).to(device)
    dummy_mask = torch.randint(0, 2, (4, 1, 224, 224)).float().to(device)

    pred  = model(dummy_img)
    loss, breakdown = loss_fn(pred, dummy_mask)

    print(f"Output shape : {pred.shape}")
    print(f"Loss         : {loss.item():.4f}  {breakdown}")
    print(f"Params       : {model.count_parameters():,}")
