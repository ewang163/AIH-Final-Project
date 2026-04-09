#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o ewang163_%j.out
#SBATCH -J ptsd_longformer

module load cuda/11.8.0-kuhf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /oscar/data/class/biol1595_2595/students/ewang163
source ptsd_env/bin/activate
python ewang163_ptsd_train_longformer.py
