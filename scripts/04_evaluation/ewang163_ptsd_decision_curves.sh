#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH -o /oscar/data/class/biol1595_2595/students/ewang163/logs/ewang163_%j.out
#SBATCH -J ptsd_dca

source /oscar/data/class/biol1595_2595/students/ewang163/ptsd_env/bin/activate
cd /oscar/data/class/biol1595_2595/students/ewang163

python /oscar/data/class/biol1595_2595/students/ewang163/scripts/04_evaluation/ewang163_ptsd_decision_curves.py
