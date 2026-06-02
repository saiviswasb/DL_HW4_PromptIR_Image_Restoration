

import argparse
import copy
import glob
import os
import random
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

class Config:
    """Default settings matching the final submitted experiment."""

    train_dir = "../Data"
    test_dir = "../Data/test/degraded"
    save_dir = "../checkpoints"
    output_dir = "../outputs"

    patch_size = 128
    batch_size = 4
    num_workers = 4
    epochs = 250
    lr = 2e-4
    min_lr = 1e-6
    warmup_epochs = 5
    grad_clip = 0.5
    ema_decay = 0.999
    dim = 64

    device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class RestorationTrainDataset(Dataset):
    """Paired rain/snow restoration dataset using the official training set."""

    def __init__(self, root_dir, patch_size=128):
        self.patch_size = patch_size
        self.pairs = []

        degraded_dir = os.path.join(root_dir, "train", "degraded")
        clean_dir = os.path.join(root_dir, "train", "clean")

        for degraded_path in sorted(glob.glob(os.path.join(degraded_dir, "*.png"))):
            filename = os.path.basename(degraded_path)
            index = filename.split("-")[1]
            clean_prefix = "rain_clean" if filename.startswith("rain-") else "snow_clean"
            clean_path = os.path.join(clean_dir, f"{clean_prefix}-{index}")
            if os.path.exists(clean_path):
                self.pairs.append((degraded_path, clean_path))

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _read_rgb(path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index):
        degraded_path, clean_path = self.pairs[index]
        degraded = self._read_rgb(degraded_path)
        clean = self._read_rgb(clean_path)

        patch = self.patch_size
        height, width = degraded.shape[:2]

        if height < patch or width < patch:
            degraded = cv2.resize(degraded, (patch, patch), interpolation=cv2.INTER_AREA)
            clean = cv2.resize(clean, (patch, patch), interpolation=cv2.INTER_AREA)
        else:
            top = random.randint(0, height - patch)
            left = random.randint(0, width - patch)
            degraded = degraded[top:top + patch, left:left + patch]
            clean = clean[top:top + patch, left:left + patch]

        if random.random() < 0.5:
            degraded = np.fliplr(degraded).copy()
            clean = np.fliplr(clean).copy()

        if random.random() < 0.5:
            degraded = np.flipud(degraded).copy()
            clean = np.flipud(clean).copy()

        rotation = random.randint(0, 3)
        if rotation > 0:
            degraded = np.rot90(degraded, rotation).copy()
            clean = np.rot90(clean, rotation).copy()

        degraded = cv2.resize(degraded, (patch, patch), interpolation=cv2.INTER_AREA)
        clean = cv2.resize(clean, (patch, patch), interpolation=cv2.INTER_AREA)

        degraded_tensor = torch.from_numpy(degraded).permute(2, 0, 1).float() / 255.0
        clean_tensor = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0
        return degraded_tensor, clean_tensor


# -----------------------------------------------------------------------------
# PromptIR-style Model
# -----------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block used as a lightweight channel attention module."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, channels = x.shape[:2]
        scale = self.fc(self.pool(x).view(batch, channels)).view(batch, channels, 1, 1)
        return x * scale


class GDFN(nn.Module):
    """Gated depth-wise feed-forward block inspired by restoration transformers."""

    def __init__(self, channels, expansion=2.66):
        super().__init__()
        hidden = int(channels * expansion)
        hidden += hidden % 2
        self.norm = LayerNorm2d(channels)
        self.proj_in = nn.Conv2d(channels, hidden * 2, 1, bias=False)
        self.depthwise = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            3,
            padding=1,
            groups=hidden * 2,
            bias=False,
        )
        self.proj_out = nn.Conv2d(hidden, channels, 1, bias=False)
        self.se = SEBlock(channels)

    def forward(self, x):
        y = self.proj_in(self.norm(x))
        y = self.depthwise(y)
        a, b = y.chunk(2, dim=1)
        y = self.proj_out(a * F.gelu(b))
        return x + self.se(y)


class PromptBlock(nn.Module):
    """Learnable prompt block to adapt features to mixed rain/snow degradation."""

    def __init__(self, channels, num_prompts=2, prompt_length=5):
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(num_prompts, prompt_length, channels) * 0.02)
        self.reduce = nn.Linear(prompt_length, 1)
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())

    def forward(self, x):
        prompt = self.reduce(self.prompt.mean(0).T).squeeze(-1).view(1, -1, 1, 1)
        return x + self.gate(x) * prompt


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2 * 4, 1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            LayerNorm2d(out_channels),
        )

    def forward(self, x):
        return self.body(x)


class PromptIR(nn.Module):
    """
    Modified PromptIR-style architecture.

    Main modifications used for the final submission:
    - DIM=64 feature width.
    - Four GDFN blocks in encoder/decoder stages.
    - Six GDFN blocks in the latent stage.
    - SE channel attention inside the feed-forward block.
    - Prompt block at the bottleneck.
    """

    def __init__(self, dim=64):
        super().__init__()
        d = dim
        self.embed = nn.Conv2d(3, d, 3, padding=1, bias=False)

        self.encoder1 = nn.Sequential(GDFN(d), GDFN(d), GDFN(d), GDFN(d))
        self.down1 = Downsample(d)
        self.encoder2 = nn.Sequential(GDFN(d * 2), GDFN(d * 2), GDFN(d * 2), GDFN(d * 2))
        self.down2 = Downsample(d * 2)

        self.latent = nn.Sequential(
            GDFN(d * 4),
            GDFN(d * 4),
            GDFN(d * 4),
            GDFN(d * 4),
            GDFN(d * 4),
            GDFN(d * 4),
        )
        self.prompt = PromptBlock(d * 4)

        self.up1 = Upsample(d * 4)
        self.fuse1 = FusionConv(d * 4, d * 2)
        self.decoder1 = nn.Sequential(GDFN(d * 2), GDFN(d * 2), GDFN(d * 2), GDFN(d * 2))

        self.up2 = Upsample(d * 2)
        self.fuse2 = FusionConv(d * 2, d)
        self.decoder2 = nn.Sequential(GDFN(d), GDFN(d), GDFN(d), GDFN(d))

        self.output = nn.Conv2d(d, 3, 3, padding=1, bias=False)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")

    def forward(self, x):
        identity = x
        x = self.embed(x)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.down1(enc1))
        latent = self.prompt(self.latent(self.down2(enc2)))
        dec1 = self.decoder1(self.fuse1(torch.cat([self.up1(latent), enc2], dim=1)))
        dec2 = self.decoder2(self.fuse2(torch.cat([self.up2(dec1), enc1], dim=1)))
        return self.output(dec2) + identity


# -----------------------------------------------------------------------------
# Loss and Metrics
# -----------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps))


class FFTLoss(nn.Module):
    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


class TotalLoss(nn.Module):
    """Hybrid restoration loss: Charbonnier + 0.1 FFT."""

    def __init__(self):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.fft = FFTLoss()

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        return self.charbonnier(pred, target) + 0.1 * self.fft(pred, target)


def calculate_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse.item() == 0:
        return 100.0
    return (20.0 * torch.log10(1.0 / torch.sqrt(mse))).item()


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------

def tta_forward(model, x):
    """Final selected 8-fold TTA: 4 rotations x horizontal flip/no flip."""
    outputs = []
    for flip in (False, True):
        for rotation in range(4):
            augmented = x.clone()
            if flip:
                augmented = torch.flip(augmented, dims=[3])
            if rotation > 0:
                augmented = torch.rot90(augmented, rotation, dims=[2, 3])

            pred = model(augmented)

            if rotation > 0:
                pred = torch.rot90(pred, -rotation, dims=[2, 3])
            if flip:
                pred = torch.flip(pred, dims=[3])

            outputs.append(pred)
    return torch.stack(outputs, dim=0).mean(0)


def list_test_images(test_dir):
    paths = []
    for idx in range(100):
        path = os.path.join(test_dir, f"{idx}.png")
        if os.path.exists(path):
            paths.append(path)
    return paths


def run_inference(model, config, output_path=None):
    if output_path is None:
        output_path = os.path.join(config.output_dir, "pred")

    os.makedirs(config.output_dir, exist_ok=True)
    image_paths = list_test_images(config.test_dir)
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {config.test_dir}")

    model.eval()
    results = {}

    with torch.no_grad():
        for path in tqdm(image_paths, desc="Inference"):
            filename = os.path.basename(path)
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Cannot read test image: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_h, original_w = image.shape[:2]

            pad_h = (16 - original_h % 16) % 16
            pad_w = (16 - original_w % 16) % 16
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

            tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
            tensor = tensor.unsqueeze(0).to(config.device)

            pred = torch.clamp(tta_forward(model, tensor), 0.0, 1.0)
            restored = pred.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
            restored = np.rint(restored).clip(0, 255).astype(np.uint8)
            restored = restored[:original_h, :original_w, :]

            results[filename] = restored.transpose(2, 0, 1)

    save_path = output_path.replace(".npz", "")
    np.savez(save_path, **results)
    sample_key = sorted(results.keys(), key=lambda x: int(x.replace(".png", "")))[0]
    print(f"Saved {len(results)} restored images -> {save_path}.npz")
    print(f"Sample output shape: {results[sample_key].shape}")


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_model(config):
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    dataset = RestorationTrainDataset(config.train_dir, patch_size=config.patch_size)
    print(f"Train pairs: {len(dataset)} | Device: {config.device}")

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )

    model = PromptIR(dim=config.dim).to(config.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = TotalLoss().to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs - config.warmup_epochs,
        eta_min=config.min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_epochs],
    )

    scaler = torch.cuda.amp.GradScaler(enabled=config.device == "cuda")

    ema_model = copy.deepcopy(model).eval()
    for param in ema_model.parameters():
        param.requires_grad = False

    best_loss = float("inf")
    ckpt_path = os.path.join(config.save_dir, "best.pth")

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.epochs}")

        for degraded, clean in pbar:
            degraded = degraded.to(config.device, non_blocking=True)
            clean = clean.to(config.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=config.device == "cuda"):
                pred = model(degraded)

            loss = criterion(pred, clean)
            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(config.ema_decay).add_(
                        model_param.data,
                        alpha=1.0 - config.ema_decay,
                    )

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix(
                loss=f"{loss.item():.5f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()
        if steps == 0:
            continue

        avg_loss = total_loss / steps
        print(f"[Epoch {epoch + 1:03d}] loss={avg_loss:.6f} lr={optimizer.param_groups[0]['lr']:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ema_model.state_dict(), ckpt_path)
            print(f"  Saved best checkpoint: {ckpt_path} | loss={best_loss:.6f}")

    model.load_state_dict(torch.load(ckpt_path, map_location=config.device, weights_only=False))
    run_inference(model, config, os.path.join(config.output_dir, "pred"))


# -----------------------------------------------------------------------------
# Additional Experiments for Report
# -----------------------------------------------------------------------------

def evaluate_subset(model, loader, config, use_tta=False):
    model.eval()
    values = []
    with torch.no_grad():
        for degraded, clean in loader:
            degraded = degraded.to(config.device)
            clean = clean.to(config.device)
            pred = tta_forward(model, degraded) if use_tta else model(degraded)
            pred = torch.clamp(pred, 0.0, 1.0)
            values.append(calculate_psnr(pred, clean))
    return float(np.mean(values))


def run_experiments(config):
    ckpt_path = os.path.join(config.save_dir, "best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    dataset = RestorationTrainDataset(config.train_dir, patch_size=config.patch_size)
    indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(indices)
    subset_indices = indices[:80]
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=2, shuffle=False, num_workers=0)

    model = PromptIR(dim=config.dim).to(config.device)
    model.load_state_dict(torch.load(ckpt_path, map_location=config.device, weights_only=False))

    psnr_no_tta = evaluate_subset(model, loader, config, use_tta=False)
    psnr_tta = evaluate_subset(model, loader, config, use_tta=True)

    charbonnier = CharbonnierLoss().to(config.device)
    fft_loss = FFTLoss().to(config.device)
    charbonnier_values = []
    hybrid_values = []

    model.eval()
    with torch.no_grad():
        for degraded, clean in loader:
            degraded = degraded.to(config.device)
            clean = clean.to(config.device)
            pred = torch.clamp(model(degraded), 0.0, 1.0)
            charb = charbonnier(pred.float(), clean.float())
            hybrid = charb + 0.1 * fft_loss(pred.float(), clean.float())
            charbonnier_values.append(charb.item())
            hybrid_values.append(hybrid.item())

    out_dir = os.path.join(config.output_dir, "experiments")
    os.makedirs(out_dir, exist_ok=True)

    result_text = (
        "Quick Additional Experiments\n"
        "============================\n"
        f"Subset size: {len(subset)} paired training samples\n"
        f"No TTA PSNR: {psnr_no_tta:.4f}\n"
        f"8-fold TTA PSNR: {psnr_tta:.4f}\n"
        f"Charbonnier Loss: {np.mean(charbonnier_values):.6f}\n"
        f"Charbonnier + 0.1 FFT Loss: {np.mean(hybrid_values):.6f}\n"
        "\nFinal public CodaBench score used in report: 29.33 PSNR\n"
        "Final selected inference setting: standard 8-fold TTA.\n"
        "Boosted 16-view TTA was tested separately but gave 29.29, so it was not selected.\n"
    )

    with open(os.path.join(out_dir, "quick_experiment_results.txt"), "w", encoding="utf-8") as file:
        file.write(result_text)

    print(result_text)
    print(f"Saved experiment summary -> {os.path.join(out_dir, 'quick_experiment_results.txt')}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Single-file PromptIR HW4 code")
    parser.add_argument("--mode", choices=["train", "infer", "experiments"], default="infer")
    parser.add_argument("--train_dir", default=Config.train_dir)
    parser.add_argument("--test_dir", default=Config.test_dir)
    parser.add_argument("--save_dir", default=Config.save_dir)
    parser.add_argument("--output_dir", default=Config.output_dir)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--patch_size", type=int, default=Config.patch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--num_workers", type=int, default=Config.num_workers)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    config.train_dir = args.train_dir
    config.test_dir = args.test_dir
    config.save_dir = args.save_dir
    config.output_dir = args.output_dir
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.patch_size = args.patch_size
    config.lr = args.lr
    config.num_workers = args.num_workers

    print(f"Mode: {args.mode}")
    print(f"Device: {config.device}")

    if args.mode == "train":
        train_model(config)
    elif args.mode == "infer":
        ckpt_path = os.path.join(config.save_dir, "best.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model = PromptIR(dim=config.dim).to(config.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=config.device, weights_only=False))
        run_inference(model, config, os.path.join(config.output_dir, "pred"))
    elif args.mode == "experiments":
        run_experiments(config)


if __name__ == "__main__":
    main()
