#!/bin/bash
#SBATCH --job-name=resshift_eval_farm
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=btulu@ufl.edu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# IMPORTANT (Batch GPU request on HiPerGator):
#SBATCH --gpus=1

# Force L4 GPU nodes (Slurm feature for NVIDIA L4 is "l4"):
#SBATCH --constraint=l4

set -euo pipefail
umask 0022

REPO="/blue/rcstudents/btulu/Projects/Image_enhancement/Diff_model/resshift_farm/ResShift"
ENV_PATH="/blue/rcstudents/btulu/.conda/envs/resshift"

CFG="$REPO/configs/farm_realsr_x4.yaml"
CKPT="$REPO/logs/farm_sr_x4/2025-12-12-18-04/ckpts/model_50000.pth"

mkdir -p "$REPO/logs"

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Start: $(date)"
echo "CWD  : $(pwd)"
echo "Partition: ${SLURM_JOB_PARTITION:-}"
echo "Account  : ${SLURM_JOB_ACCOUNT:-}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES(before): ${CUDA_VISIBLE_DEVICES:-}"

module purge
module load conda/25.7.0
module list

echo "=== scontrol show job (key fields) ==="
scontrol show job "$SLURM_JOB_ID" | egrep -i "JobId=|Account=|Partition=|NodeList=|TRES=|Gres=|Features=|Constraint=" || true

echo "=== nvidia-smi ==="
nvidia-smi -L || true
nvidia-smi || true

# ---- activate conda env (NO ~/.bashrc) ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

echo "=== Python/Torch GPU check ==="
python - <<'PY'
import os, torch
print("Torch:", torch.__version__)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA available:", torch.cuda.is_available())
print("CUDA device_count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA not available (GPU not allocated/visible).")
print("GPU name:", torch.cuda.get_device_name(0))
PY

cd "$REPO"
[ -f "$CFG" ]  || { echo "ERROR: Missing config: $CFG"; exit 1; }
[ -f "$CKPT" ] || { echo "ERROR: Missing checkpoint: $CKPT"; exit 1; }

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "=== Running FIELD (big .tif) ==="
python farm_resshift_infer_eval.py \
  --hr_dir testdata/farm/field \
  --out_root results/farm_field_eval \
  --cfg_path "$CFG" \
  --ckpt_path "$CKPT" \
  --scale 4 \
  --chop_size 256 \
  --chop_stride 224 \
  --use_amp \
  --save_bicubic \
  --num_vis 10

echo "=== Running PLOT (.jpg) ==="
python farm_resshift_infer_eval.py \
  --hr_dir testdata/farm/plot \
  --out_root results/farm_plot_eval \
  --cfg_path "$CFG" \
  --ckpt_path "$CKPT" \
  --scale 4 \
  --chop_size 512 \
  --chop_stride 448 \
  --use_amp \
  --save_bicubic \
  --num_vis 10

echo "End: $(date)"
echo "Outputs:"
echo "  $REPO/results/farm_field_eval"
echo "  $REPO/results/farm_plot_eval"
