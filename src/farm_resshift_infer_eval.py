#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end evaluation for a fine-tuned ResShift SR model:

- Reads HR images in a folder (your "GT")
- Crops HR to multiple of scale
- Downscales by scale to create LR (RAW)
- Pads LR (PAD) to multiples of lq_size (e.g., 64) using REPLICATE padding (safe)
- Runs ResShiftSampler inference on padded LR
- Crops SR outputs back to original HR size
- Computes PSNR / SSIM / (optional) LPIPS vs original HR
- Saves:
  - LR_x4_raw/        (raw downscaled)
  - LR_x4_padded/     (padded LR fed to ResShift)
  - enhanced_ResShift/ (raw outputs from sampler)
  - enhanced_ResShift_cropped/ (final SR cropped to HR size)
  - bicubic_up/       (optional)
  - metrics/metrics.csv + metrics/summary.json
  - figures/ (optional) + plots/ (optional)

Important fix:
ResShift uses reflect padding internally; reflect padding fails when pad >= input dim.
We pre-pad LR to lq_size multiples using replicate padding.
"""

import os
import sys
import math
import json
import csv
import random
import argparse
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np

import torch
import torch.nn.functional as F

# -----------------------------
# Optional matplotlib (skip figures/plots if missing)
# -----------------------------
_HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception as e:
    print(f"[WARN] matplotlib not available ({e}); plots/figures will be skipped.")

# LPIPS optional
try:
    import lpips
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False


# -----------------------------
# IO helpers (cv2 preferred; PIL fallback)
# -----------------------------
def _try_import_cv2():
    try:
        import cv2
        return cv2
    except Exception:
        return None

def _try_import_pil():
    try:
        from PIL import Image
        return Image
    except Exception:
        return None

cv2 = _try_import_cv2()
PIL_Image = _try_import_pil()


def read_image_rgb_float01(path: Path) -> np.ndarray:
    """
    Returns RGB float32 in [0,1], shape (H,W,3).
    Supports png/jpg/tif/tiff (depends on backend).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    if cv2 is not None:
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if im is not None:
            # grayscale -> RGB
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            # BGRA -> BGR
            elif im.ndim == 3 and im.shape[2] == 4:
                im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)

            # BGR -> RGB
            if im.ndim == 3 and im.shape[2] == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            if im.dtype == np.uint16:
                im = im.astype(np.float32) / 65535.0
            elif im.dtype == np.uint8:
                im = im.astype(np.float32) / 255.0
            else:
                im = im.astype(np.float32)
                if im.max() > 1.5:
                    im = im / 255.0

            return np.clip(im, 0.0, 1.0)

    if PIL_Image is None:
        raise RuntimeError("Could not read images: neither cv2 nor PIL is available.")

    with PIL_Image.open(str(path)) as img:
        img = img.convert("RGB")
        im = np.array(img).astype(np.float32) / 255.0
        return np.clip(im, 0.0, 1.0)


def write_image_rgb_float01(path: Path, im_rgb01: np.ndarray, out_ext: str = ".png") -> Path:
    """
    Save RGB float [0,1] to disk (8-bit). Returns actual saved path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im = np.clip(im_rgb01, 0.0, 1.0)
    im8 = (im * 255.0 + 0.5).astype(np.uint8)

    out_path = path.with_suffix(out_ext)

    if cv2 is not None:
        bgr = cv2.cvtColor(im8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr)
        return out_path

    if PIL_Image is None:
        raise RuntimeError("Could not write images: neither cv2 nor PIL is available.")

    PIL_Image.fromarray(im8).save(str(out_path))
    return out_path


def resize_rgb(im_rgb01: np.ndarray, new_hw: Tuple[int, int], mode: str) -> np.ndarray:
    """
    Resize RGB image to (new_h, new_w). mode in {"area","bicubic"}.
    """
    new_h, new_w = int(new_hw[0]), int(new_hw[1])
    im = np.clip(im_rgb01, 0.0, 1.0)

    if cv2 is not None:
        interp = cv2.INTER_AREA if mode == "area" else cv2.INTER_CUBIC
        im8 = (im * 255.0 + 0.5).astype(np.uint8)
        out8 = cv2.resize(im8, (new_w, new_h), interpolation=interp)
        out = out8.astype(np.float32) / 255.0
        return np.clip(out, 0.0, 1.0)

    # torch fallback
    t = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    if mode == "bicubic":
        t = F.interpolate(t, size=(new_h, new_w), mode="bicubic", align_corners=False)
    else:
        t = F.interpolate(t, size=(new_h, new_w), mode="area")
    out = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(out, 0.0, 1.0)


def pad_right_bottom_to_multiple(im_rgb01: np.ndarray, multiple: int) -> Tuple[np.ndarray, int, int]:
    """
    Pad on bottom/right so (H,W) become multiples of `multiple`.
    Uses replicate padding (safe even when pad >= input dimension).
    Returns: padded_image, pad_h, pad_w
    """
    h, w = im_rgb01.shape[:2]
    multiple = int(multiple)
    if multiple <= 0:
        return im_rgb01, 0, 0

    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple

    # also ensure at least multiple in each dim (important when h < multiple)
    if h + pad_h < multiple:
        pad_h = multiple - h
    if w + pad_w < multiple:
        pad_w = multiple - w

    if pad_h == 0 and pad_w == 0:
        return im_rgb01, 0, 0

    t = torch.from_numpy(np.clip(im_rgb01, 0.0, 1.0)).permute(2, 0, 1).unsqueeze(0).float()
    t = F.pad(t, pad=(0, pad_w, 0, pad_h), mode="replicate")  # (L,R,T,B) for 4D is (wL,wR,hT,hB)
    out = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(out, 0.0, 1.0), int(pad_h), int(pad_w)


# -----------------------------
# Metrics
# -----------------------------
def psnr_np(x: np.ndarray, y: np.ndarray, data_range: float = 1.0) -> float:
    x = np.clip(x, 0.0, data_range).astype(np.float32)
    y = np.clip(y, 0.0, data_range).astype(np.float32)
    mse = float(np.mean((x - y) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def _gaussian_window(window_size: int = 11, sigma: float = 1.5, device="cpu") -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)  # 1,1,ws,ws
    return w


def ssim_torch(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0,
               window_size: int = 11, sigma: float = 1.5) -> float:
    """
    img1,img2: (1,3,H,W) in [0,1]
    returns mean SSIM over channels
    """
    device = img1.device
    w = _gaussian_window(window_size, sigma, device=device)
    c = img1.shape[1]
    w = w.repeat(c, 1, 1, 1)  # (C,1,ws,ws)

    pad = window_size // 2
    mu1 = F.conv2d(img1, w, padding=pad, groups=c)
    mu2 = F.conv2d(img2, w, padding=pad, groups=c)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, w, padding=pad, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, w, padding=pad, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, w, padding=pad, groups=c) - mu12

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12)
    return float(ssim_map.mean().detach().cpu().item())


@torch.no_grad()
def lpips_score(lpips_net, x_rgb01: np.ndarray, y_rgb01: np.ndarray,
                device: torch.device, resize_to: int = 256) -> float:
    """
    LPIPS expects [-1,1], NCHW.
    For huge images, resize_to keeps it fast (default 256). Set resize_to=0 for full size.
    """
    x = torch.from_numpy(np.clip(x_rgb01, 0, 1)).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(np.clip(y_rgb01, 0, 1)).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)

    if resize_to and (x.shape[-1] != resize_to or x.shape[-2] != resize_to):
        x = F.interpolate(x, size=(resize_to, resize_to), mode="bilinear", align_corners=False)
        y = F.interpolate(y, size=(resize_to, resize_to), mode="bilinear", align_corners=False)

    x = x * 2.0 - 1.0
    y = y * 2.0 - 1.0
    d = lpips_net(x, y)
    return float(d.mean().detach().cpu().item())


# -----------------------------
# ResShift inference wrapper
# -----------------------------
def build_resshift_sampler(cfg_path: Path, ckpt_path: Path, scale: int,
                           chop_size: int, chop_stride: int, chop_bs: int,
                           use_amp: bool, seed: int):
    """
    Uses ResShiftSampler from ResShift repo.
    """
    from omegaconf import OmegaConf

    # ensure repo root import
    sys.path.insert(0, str(Path.cwd()))
    from sampler import ResShiftSampler

    cfg = OmegaConf.load(str(cfg_path))

    cfg.model.ckpt_path = str(ckpt_path)
    cfg.diffusion.params.sf = int(scale)

    padding_offset = int(cfg.model.params.get("lq_size", 64))

    sampler = ResShiftSampler(
        cfg,
        sf=int(scale),
        chop_size=int(chop_size),
        chop_stride=int(chop_stride),
        chop_bs=int(chop_bs),
        use_amp=bool(use_amp),
        seed=int(seed),
        padding_offset=padding_offset,
    )
    return sampler


def find_images(input_dir: Path, exts: Tuple[str, ...]) -> List[Path]:
    files = []
    for e in exts:
        files.extend(sorted(input_dir.rglob(f"*{e}")))
    seen = set()
    out = []
    for p in files:
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            out.append(p)
    return out


def crop_to_multiple(im: np.ndarray, scale: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = im.shape[:2]
    hc = h - (h % scale)
    wc = w - (w % scale)
    return im[:hc, :wc, :], (hc, wc)


def match_size(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h = min(ha, hb)
    w = min(wa, wb)
    return a[:h, :w, :], b[:h, :w, :]


def find_first_output(sr_dir: Path, stem: str, prefer_exts: List[str]) -> Optional[Path]:
    # 1) exact matches in sr_dir
    for ext in prefer_exts:
        p = sr_dir / f"{stem}{ext}"
        if p.exists():
            return p

    # 2) recursive search
    for ext in prefer_exts:
        cands = list(sr_dir.rglob(f"{stem}{ext}"))
        if cands:
            return sorted(cands)[0]

    # 3) any file starting with stem
    cands = list(sr_dir.rglob(f"{stem}*"))
    cands = [c for c in cands if c.is_file()]
    if cands:
        return sorted(cands)[0]
    return None


def make_comparison_figure(hr, lr_up, bic, sr, out_path: Path, title: str, crop_size: int = 512):
    if not _HAS_MPL:
        return
    H, W = hr.shape[:2]
    cs = int(min(crop_size, H, W))
    y0 = max(0, (H - cs) // 2)
    x0 = max(0, (W - cs) // 2)

    hr_c = hr[y0:y0+cs, x0:x0+cs]
    lr_c = lr_up[y0:y0+cs, x0:x0+cs]
    bic_c = bic[y0:y0+cs, x0:x0+cs]
    sr_c = sr[y0:y0+cs, x0:x0+cs]

    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(2, 2, 1); ax1.imshow(hr_c); ax1.set_title("HR (GT)"); ax1.axis("off")
    ax2 = plt.subplot(2, 2, 2); ax2.imshow(lr_c); ax2.set_title("LR (downscale×4) upsampled"); ax2.axis("off")
    ax3 = plt.subplot(2, 2, 3); ax3.imshow(bic_c); ax3.set_title("Bicubic baseline"); ax3.axis("off")
    ax4 = plt.subplot(2, 2, 4); ax4.imshow(sr_c); ax4.set_title("ResShift SR"); ax4.axis("off")
    plt.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200)
    plt.close(fig)


def plot_histograms(rows: List[Dict], out_dir: Path):
    if not _HAS_MPL:
        return
    psnr_b = [r["psnr_bicubic"] for r in rows if r.get("psnr_bicubic") is not None]
    psnr_s = [r["psnr_resshift"] for r in rows if r.get("psnr_resshift") is not None]
    dpsnr  = [r["delta_psnr"] for r in rows if r.get("delta_psnr") is not None]

    out_dir.mkdir(parents=True, exist_ok=True)

    if psnr_b and psnr_s:
        plt.figure()
        plt.hist(psnr_b, bins=20, alpha=0.7, label="Bicubic")
        plt.hist(psnr_s, bins=20, alpha=0.7, label="ResShift")
        plt.xlabel("PSNR (dB)")
        plt.ylabel("Count")
        plt.legend()
        plt.title("PSNR distribution")
        plt.tight_layout()
        plt.savefig(str(out_dir / "psnr_hist.png"), dpi=200)
        plt.close()

    if dpsnr:
        plt.figure()
        plt.hist(dpsnr, bins=20)
        plt.xlabel("ΔPSNR (ResShift - Bicubic) (dB)")
        plt.ylabel("Count")
        plt.title("ΔPSNR distribution")
        plt.tight_layout()
        plt.savefig(str(out_dir / "delta_psnr_hist.png"), dpi=200)
        plt.close()


def write_csv(rows: List[Dict], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary(rows: List[Dict], out_json: Path):
    def _stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        arr = np.array(vals, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": int(arr.size),
        }

    summary = {
        "psnr_bicubic": _stats([r.get("psnr_bicubic") for r in rows]),
        "psnr_resshift": _stats([r.get("psnr_resshift") for r in rows]),
        "delta_psnr": _stats([r.get("delta_psnr") for r in rows]),
        "ssim_bicubic": _stats([r.get("ssim_bicubic") for r in rows]),
        "ssim_resshift": _stats([r.get("ssim_resshift") for r in rows]),
        "delta_ssim": _stats([r.get("delta_ssim") for r in rows]),
        "lpips_bicubic": _stats([r.get("lpips_bicubic") for r in rows]),
        "lpips_resshift": _stats([r.get("lpips_resshift") for r in rows]),
        "delta_lpips": _stats([r.get("delta_lpips") for r in rows]),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr_dir", type=str, required=True, help="Folder with original HR images (GT).")
    ap.add_argument("--out_root", type=str, required=True, help="Output root folder.")
    ap.add_argument("--cfg_path", type=str, required=True, help="ResShift config yaml used in training.")
    ap.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint .pth (e.g., model_50000.pth).")
    ap.add_argument("--scale", type=int, default=4, help="Downscale factor (must match training).")
    ap.add_argument("--exts", type=str, default=".png,.jpg,.jpeg,.tif,.tiff", help="Extensions to include.")
    ap.add_argument("--max_images", type=int, default=0, help="0=all, else limit number of images.")
    ap.add_argument("--seed", type=int, default=12345)

    # ResShift chopping params
    ap.add_argument("--chop_size", type=int, default=512)
    ap.add_argument("--chop_stride", type=int, default=448)
    ap.add_argument("--chop_bs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=1, help="Batch size for sampler inference dataloader.")
    ap.add_argument("--use_amp", action="store_true", help="Use AMP for inference.")

    # LPIPS
    ap.add_argument("--lpips_resize", type=int, default=256, help="Resize to NxN before LPIPS. 0=full size.")
    ap.add_argument("--no_lpips", action="store_true", help="Disable LPIPS even if installed.")

    # Saving & visualization
    ap.add_argument("--out_ext", type=str, default=".png", help="Output image extension for saved files.")
    ap.add_argument("--save_bicubic", action="store_true", help="Save bicubic baseline images.")
    ap.add_argument("--num_vis", type=int, default=10, help="How many comparison figures to save.")
    ap.add_argument("--vis_crop_size", type=int, default=512, help="Crop size for visualization panels.")
    ap.add_argument("--skip_inference", action="store_true", help="Skip inference (assumes SR already exists).")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    hr_dir = Path(args.hr_dir)
    out_root = Path(args.out_root)
    cfg_path = Path(args.cfg_path)
    ckpt_path = Path(args.ckpt_path)

    if not hr_dir.exists():
        raise FileNotFoundError(str(hr_dir))
    if not cfg_path.exists():
        raise FileNotFoundError(str(cfg_path))
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    # Read lq_size (padding offset) from config
    from omegaconf import OmegaConf
    cfg0 = OmegaConf.load(str(cfg_path))
    lq_size = int(cfg0.model.params.get("lq_size", 64))
    print(f"[INFO] lq_size (padding_offset) = {lq_size}")

    lr_raw_dir = out_root / "LR_x4_raw"
    lr_pad_dir = out_root / "LR_x4_padded"
    sr_dir = out_root / "enhanced_ResShift"
    sr_crop_dir = out_root / "enhanced_ResShift_cropped"
    bic_dir = out_root / "bicubic_up"
    metrics_dir = out_root / "metrics"
    figs_dir = out_root / "figures"
    plots_dir = out_root / "plots"

    # Make dirs
    for d in [lr_raw_dir, lr_pad_dir, sr_dir, sr_crop_dir, metrics_dir, figs_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)
    if args.save_bicubic:
        bic_dir.mkdir(parents=True, exist_ok=True)

    # Clean old LR padded/raw (avoid stale files)
    for d in [lr_raw_dir, lr_pad_dir]:
        for f in d.glob("*"):
            if f.is_file():
                f.unlink()

    exts = tuple([e.strip().lower() for e in args.exts.split(",") if e.strip()])
    files = find_images(hr_dir, exts)
    if args.max_images and args.max_images > 0:
        files = files[:args.max_images]
    if not files:
        print(f"[ERROR] No images found in {hr_dir} with exts={exts}")
        return

    print(f"[INFO] Found {len(files)} images in {hr_dir}")
    print(f"[INFO] Writing LR raw   -> {lr_raw_dir}")
    print(f"[INFO] Writing LR padded-> {lr_pad_dir}")

    # Build LR inputs + mapping
    mapping: List[Dict] = []
    for idx, p in enumerate(files, 1):
        try:
            hr = read_image_rgb_float01(p)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
            continue

        hr, (hc, wc) = crop_to_multiple(hr, args.scale)
        if hc <= 0 or wc <= 0:
            print(f"[WARN] Image too small after cropping: {p}")
            continue

        # LR raw: exactly downscale by scale
        lr_raw = resize_rgb(hr, (hc // args.scale, wc // args.scale), mode="area")
        lr_raw_path = write_image_rgb_float01(lr_raw_dir / p.stem, lr_raw, out_ext=args.out_ext)

        # LR padded: replicate pad to multiples of lq_size
        lr_pad, pad_h, pad_w = pad_right_bottom_to_multiple(lr_raw, lq_size)
        lr_pad_path = write_image_rgb_float01(lr_pad_dir / p.stem, lr_pad, out_ext=args.out_ext)

        mapping.append({
            "stem": p.stem,
            "hr_path": str(p),
            "hr_h": int(hc),
            "hr_w": int(wc),
            "lr_raw_path": str(lr_raw_path),
            "lr_pad_path": str(lr_pad_path),
            "lr_pad_h": int(pad_h),
            "lr_pad_w": int(pad_w),
        })

        if idx % 25 == 0 or idx == len(files):
            print(f"  - LR prepared: {idx}/{len(files)}")

    if not mapping:
        print("[ERROR] No LR inputs were created.")
        return

    # Inference
    if not args.skip_inference:
        print(f"[INFO] Running ResShift inference -> {sr_dir}")

        # Clear sr_dir to avoid mixing old/new
        for f in sr_dir.rglob("*"):
            if f.is_file():
                f.unlink()

        sampler = build_resshift_sampler(
            cfg_path=cfg_path,
            ckpt_path=ckpt_path,
            scale=args.scale,
            chop_size=args.chop_size,
            chop_stride=args.chop_stride,
            chop_bs=args.chop_bs,
            use_amp=args.use_amp,
            seed=args.seed,
        )

        sampler.inference(
            str(lr_pad_dir),
            str(sr_dir),
            mask_path=None,
            bs=int(args.bs),
            noise_repeat=False,
        )

        del sampler
        torch.cuda.empty_cache()
    else:
        print("[INFO] Skipping inference (using existing SR outputs).")

    # Crop SR outputs to HR size (top-left crop)
    print(f"[INFO] Cropping SR outputs -> {sr_crop_dir}")
    for f in sr_crop_dir.glob("*"):
        if f.is_file():
            f.unlink()

    prefer_exts = [args.out_ext, ".png", ".jpg", ".jpeg", ".tif", ".tiff"]

    for m in mapping:
        stem = m["stem"]
        sr_path = find_first_output(sr_dir, stem, prefer_exts)
        if sr_path is None:
            print(f"[WARN] Missing SR output for {stem} (searched {sr_dir})")
            m["sr_crop_path"] = None
            continue

        sr = read_image_rgb_float01(sr_path)
        hr_h, hr_w = m["hr_h"], m["hr_w"]

        sr_c = sr[:hr_h, :hr_w, :]
        out_final = write_image_rgb_float01(sr_crop_dir / stem, sr_c, out_ext=args.out_ext)
        m["sr_crop_path"] = str(out_final)

    # LPIPS init (optional)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_net = None
    if (not args.no_lpips) and _HAS_LPIPS:
        try:
            lpips_net = lpips.LPIPS(net="vgg").to(device).eval()
        except Exception as e:
            print(f"[WARN] Could not init LPIPS: {e}")
            lpips_net = None
    elif not _HAS_LPIPS and not args.no_lpips:
        print("[WARN] lpips package not available; LPIPS will be skipped.")

    # Evaluate
    print("[INFO] Computing metrics (PSNR/SSIM/LPIPS) + saving outputs...")
    rows: List[Dict] = []
    vis_count = 0

    for i, m in enumerate(mapping, 1):
        stem = m["stem"]
        hr_path = Path(m["hr_path"])
        lr_raw_path = Path(m["lr_raw_path"])
        sr_crop_path = m.get("sr_crop_path", None)

        if sr_crop_path is None:
            continue

        # Read HR and crop to stored size
        hr = read_image_rgb_float01(hr_path)
        hr = hr[:m["hr_h"], :m["hr_w"], :]

        # Read LR raw
        lr_raw = read_image_rgb_float01(lr_raw_path)

        # Bicubic baseline from LR raw
        bic = resize_rgb(lr_raw, (m["hr_h"], m["hr_w"]), mode="bicubic")

        # Read SR cropped
        sr = read_image_rgb_float01(Path(sr_crop_path))

        # Align sizes robustly
        hr2, bic2 = match_size(hr, bic)
        hr3, sr2 = match_size(hr, sr)
        # align bic to hr3
        bic3, _ = match_size(bic2, hr3)
        # final align
        hr_f, bic_f = match_size(hr3, bic3)
        hr_f, sr_f = match_size(hr_f, sr2)

        # Metrics
        psnr_b = psnr_np(bic_f, hr_f)
        psnr_s = psnr_np(sr_f, hr_f)

        t_hr = torch.from_numpy(hr_f).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        t_bi = torch.from_numpy(bic_f).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        t_sr = torch.from_numpy(sr_f).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)

        ssim_b = ssim_torch(t_bi, t_hr)
        ssim_s = ssim_torch(t_sr, t_hr)

        lp_b = None
        lp_s = None
        if lpips_net is not None:
            lp_b = lpips_score(lpips_net, bic_f, hr_f, device=device, resize_to=int(args.lpips_resize))
            lp_s = lpips_score(lpips_net, sr_f, hr_f, device=device, resize_to=int(args.lpips_resize))

        row = {
            "stem": stem,
            "hr_path": str(hr_path),
            "lr_raw_path": str(lr_raw_path),
            "lr_pad_path": str(m["lr_pad_path"]),
            "sr_raw_dir": str(sr_dir),
            "sr_crop_path": str(sr_crop_path),
            "H": int(hr_f.shape[0]),
            "W": int(hr_f.shape[1]),
            "lr_pad_h": int(m["lr_pad_h"]),
            "lr_pad_w": int(m["lr_pad_w"]),
            "psnr_bicubic": float(psnr_b),
            "psnr_resshift": float(psnr_s),
            "delta_psnr": float(psnr_s - psnr_b),
            "ssim_bicubic": float(ssim_b),
            "ssim_resshift": float(ssim_s),
            "delta_ssim": float(ssim_s - ssim_b),
            "lpips_bicubic": None if lp_b is None else float(lp_b),
            "lpips_resshift": None if lp_s is None else float(lp_s),
            "delta_lpips": None if (lp_b is None or lp_s is None) else float(lp_s - lp_b),
        }
        rows.append(row)

        # Save bicubic images (cropped size)
        if args.save_bicubic:
            write_image_rgb_float01(bic_dir / stem, bic_f, out_ext=args.out_ext)

        # Save a few comparison figures
        if vis_count < args.num_vis:
            lr_up = resize_rgb(lr_raw, (m["hr_h"], m["hr_w"]), mode="bicubic")
            fig_path = figs_dir / f"{stem}_panel.png"
            make_comparison_figure(hr_f, lr_up, bic_f, sr_f, fig_path, title=stem, crop_size=args.vis_crop_size)
            vis_count += 1

        if i % 25 == 0 or i == len(mapping):
            print(f"  - Evaluated: {i}/{len(mapping)}")

    # Save metrics
    out_csv = metrics_dir / "metrics.csv"
    out_summary = metrics_dir / "summary.json"
    write_csv(rows, out_csv)
    write_summary(rows, out_summary)
    plot_histograms(rows, plots_dir)

    print("====================================")
    print(f"[DONE] LR raw            : {lr_raw_dir}")
    print(f"[DONE] LR padded (input) : {lr_pad_dir}")
    print(f"[DONE] SR raw outputs    : {sr_dir}")
    print(f"[DONE] SR cropped outputs: {sr_crop_dir}")
    if args.save_bicubic:
        print(f"[DONE] Bicubic outputs   : {bic_dir}")
    print(f"[DONE] Metrics CSV       : {out_csv}")
    print(f"[DONE] Summary JSON      : {out_summary}")
    if _HAS_MPL:
        print(f"[DONE] Figures           : {figs_dir}")
        print(f"[DONE] Plots             : {plots_dir}")
    else:
        print("[DONE] (matplotlib missing) skipped figures/plots")


if __name__ == "__main__":
    main()
