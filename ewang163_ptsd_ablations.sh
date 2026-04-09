#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH -o ewang163_%j.out
#SBATCH -J ptsd_ablations

source /oscar/data/class/biol1595_2595/students/ewang163/ptsd_env/bin/activate
python /oscar/data/class/biol1595_2595/students/ewang163/ewang163_ptsd_ablations.py
