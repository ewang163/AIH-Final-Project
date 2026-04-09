#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH -o ewang163_%j.out
#SBATCH -J ptsd_errors

source /oscar/data/class/biol1595_2595/students/ewang163/ptsd_env/bin/activate
cd /oscar/data/class/biol1595_2595/students/ewang163

python ewang163_ptsd_error_analysis.py
