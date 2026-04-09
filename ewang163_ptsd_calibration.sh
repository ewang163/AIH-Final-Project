#!/bin/bash
#SBATCH -p batch
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH -o ewang163_%j.out
#SBATCH -J ptsd_calib

source /oscar/data/class/biol1595_2595/students/ewang163/ptsd_env/bin/activate
cd /oscar/data/class/biol1595_2595/students/ewang163

python ewang163_ptsd_calibration.py
