# Remote-Sensing-Generative-Restoration

[![Framework](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Active-green)]()

> **A repository for exploring and implementing Generative AI techniques (SRGAN, ESRGAN, Diffusion) to restore, super-resolve, and enhance satellite and drone imagery.**

---

## 🛰️ Project Overview

This project addresses common degradation issues in aerial and satellite imagery—such as low resolution, atmospheric haze, and sensor noise. It leverages state-of-the-art Generative AI architectures to improve image quality for downstream agricultural and environmental analysis.

The goal is to benchmark and optimize different generative models to determine the best trade-off between perceptual quality and computational efficiency for remote sensing tasks.

## 🧠 Implemented Models

This repository contains implementations and adaptations of the following architectures:

* **SRGAN (Super-Resolution GAN):** Baseline adversarial network for photorealistic upscaling.
* **ESRGAN (Enhanced SRGAN):** Improved perceptual quality using Residual-in-Residual Dense Blocks (RRDB).
* **Real-ESRGAN:** Optimized for restoring real-world degradation using pure synthetic data training.
* **Diffusion Models:** Latent Diffusion Models (LDMs) for high-fidelity image restoration and conditional generation.

## 🧪 Experiments & Methodology

The models are currently being benchmarked on:
1.  **Synthetic Degradation:** High-quality drone footage artificially downsampled/blurred for training.
2.  **Real-World Testing:** Application of trained weights to raw Sentinel-2 satellite tiles and noisy agricultural drone imagery.

## 💻 Usage

### Environment Setup
```bash
# Clone the repository
git clone [https://github.com/YourUsername/Remote-Sensing-Generative-Restoration.git](https://github.com/YourUsername/Remote-Sensing-Generative-Restoration.git)
cd Remote-Sensing-Generative-Restoration

# Create environment
conda create -n rs_genai python=3.9
conda activate rs_genai

# Install dependencies
pip install -r requirements.txt
