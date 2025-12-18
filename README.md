# Remote-Sensing-Generative-Restoration

[![Framework](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Active-green)]()

> Generative AI (SRGAN, ESRGAN/Real-ESRGAN, diffusion) for restoring, super-resolving, and enhancing satellite and drone imagery.

---

## Project Overview
This project tackles common degradation in aerial and satellite imagery - low resolution, haze, and sensor noise - using state-of-the-art generative models. The goal is to benchmark quality vs. efficiency for remote-sensing tasks in agriculture and environmental monitoring.

## Implemented Models
- SRGAN: Baseline adversarial super-resolution.
- ESRGAN (RRDB): Perceptual-quality improvements with RRDB blocks.
- Real-ESRGAN: Restoration tuned for real-world degradations.
- Diffusion (LDM): High-fidelity restoration and conditional generation.

## Experiments & Methodology
1. Synthetic Degradation: Downsampled/blurred high-quality drone footage for training.
2. Real-World Testing: Raw Sentinel-2 tiles and noisy agricultural drone imagery.

## Environment Setup
```bash
git clone https://github.com/YourUsername/Remote-Sensing-Generative-Restoration.git
cd Remote-Sensing-Generative-Restoration

# Python 3.9 recommended
python -m venv .venv
# On Windows
.\.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

pip install torch>=2.0 realesrgan basicsr opencv-python pillow tqdm
```

## Usage (Real-ESRGAN inference)
```bash
python src/infer_farm_slurm.py \
  --ckpt path/to/RealESRGAN_x4plus.pth \
  --input path/to/images_or_dir \
  --outdir outputs/x4 \
  --tile 512 \
  --tile-pad 10 \
  --pre-pad 0 \
  --outscale 4.0 \
  --num-shards 1 \
  --shard-idx 0
```
- num-shards/shard-idx let you split work across jobs/GPUs.
- Outputs save as `<name>_x4.png` in `--outdir`.
- If you hit GPU OOM, lower `--tile` (e.g., 256 or 128); uses half precision when CUDA is available.

## Checkpoints
Download Real-ESRGAN weights (e.g., `RealESRGAN_x4plus.pth`) from the official repo and point `--ckpt` to that path. Add other checkpoints here as you include additional models.

## Data Notes
- Supported formats: png, jpg, jpeg, bmp, tif, tiff, webp.
- Grayscale images are auto-converted to RGB; alpha channels are dropped.
- Keep inputs in a directory to batch-process; individual file paths also work.

## Project Structure
- src/infer_farm_slurm.py - Real-ESRGAN inference with tiling and sharding.
- (Add training/experiment scripts here as they are added.)

## Roadmap / TODO
- Add training scripts and experiment configs.
- Document diffusion-based restoration pipeline.
- Provide example before/after results and benchmark metrics.

## License and Citation
Add your license terms and a citation/BibTeX entry so others know how to use and reference this work.
