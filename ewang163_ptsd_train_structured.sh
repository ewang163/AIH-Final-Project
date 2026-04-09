#!/bin/bash
#SBATCH -p batch
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH -o ewang163_%j.out

source /oscar/data/class/biol1595_2595/students/ewang163/ptsd_env/bin/activate
python /oscar/data/class/biol1595_2595/students/ewang163/ewang163_ptsd_train_structured.py
