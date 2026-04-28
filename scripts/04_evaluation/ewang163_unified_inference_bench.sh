#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH -J ewang163_unified_bench
#SBATCH -o /oscar/data/class/biol1595_2595/students/ewang163/logs/ewang163_unified_bench_%j.out

set -euo pipefail

source /oscar/data/class/biol1595_2595/students/ewang163/ptsd_env/bin/activate
cd /oscar/data/class/biol1595_2595/students/ewang163

echo "=== JOB START ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPUs:     $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

python -u scripts/04_evaluation/ewang163_unified_inference_bench.py

echo "=== JOB END ==="
