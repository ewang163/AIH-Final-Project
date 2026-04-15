#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o /oscar/data/class/biol1595_2595/students/ewang163/logs/ewang163_%j.out
#SBATCH -J ptsd_bioclinbert

source /oscar/data/class/biol1595_2595/students/ewang163/ptsd_env/bin/activate
cd /oscar/data/class/biol1595_2595/students/ewang163

python /oscar/data/class/biol1595_2595/students/ewang163/scripts/03_training/ewang163_ptsd_train_bioclinbert.py
