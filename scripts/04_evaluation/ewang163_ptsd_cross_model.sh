#!/bin/bash
#SBATCH -p batch
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH -o /oscar/data/class/biol1595_2595/students/ewang163/logs/ewang163_cross_model_%j.out

source /oscar/data/class/biol1595_2595/students/ewang163/ptsd_env/bin/activate
cd /oscar/data/class/biol1595_2595/students/ewang163
python scripts/04_evaluation/ewang163_ptsd_cross_model.py
