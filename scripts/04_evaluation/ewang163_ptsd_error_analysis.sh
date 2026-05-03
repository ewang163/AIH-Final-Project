#!/bin/bash
#SBATCH -p batch
#SBATCH --mem=8G
#SBATCH --time=0:15:00
#SBATCH -o /oscar/data/class/biol1595_2595/students/ewang163/logs/ewang163_error_analysis_%j.out
#SBATCH -J ptsd_errors

source /oscar/data/class/biol1595_2595/students/ewang163/ptsd_env/bin/activate
cd /oscar/data/class/biol1595_2595/students/ewang163

python scripts/04_evaluation/ewang163_ptsd_error_analysis.py
