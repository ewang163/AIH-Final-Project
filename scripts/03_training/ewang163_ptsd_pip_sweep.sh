#!/bin/bash
# ewang163_ptsd_pip_sweep.sh
# ==========================
# Fix 2: Sweep class prior pi_p across 7 values, submitting each as a
# separate SLURM GPU job.  Each job trains a Longformer with a different
# pi_p and saves its checkpoint to a suffixed directory.
#
# After all jobs complete, run ewang163_ptsd_pip_sweep_eval.py to
# aggregate results and pick the best pi_p by proxy validation AUC.
#
# USAGE:
#     bash scripts/03_training/ewang163_ptsd_pip_sweep.sh
#
# This script submits 7 SLURM jobs and prints their job IDs.

SCRIPT_DIR="/oscar/data/class/biol1595_2595/students/ewang163/scripts/03_training"
LOG_DIR="/oscar/data/class/biol1595_2595/students/ewang163/logs"

PI_VALUES="0.05 0.08 0.10 0.12 0.15 0.20 0.25"

for PI in $PI_VALUES; do
    SUFFIX="_pip$(echo $PI | tr -d '.')"
    echo "Submitting pi_p=${PI} (suffix=${SUFFIX}) ..."

    sbatch \
        --partition=gpu \
        --gres=gpu:1 \
        --mem=32G \
        --time=12:00:00 \
        --job-name="lf_pip${PI}" \
        --output="${LOG_DIR}/ewang163_pip_sweep${SUFFIX}_%j.out" \
        --wrap="source /oscar/data/class/biol1595_2595/students/ewang163/ptsd_env/bin/activate && python ${SCRIPT_DIR}/ewang163_ptsd_train_longformer.py --pi_p ${PI} --output_suffix ${SUFFIX}"
done

echo ""
echo "All 7 sweep jobs submitted. Monitor with: squeue -u ewang163"
echo "After completion, run: python scripts/03_training/ewang163_ptsd_pip_sweep_eval.py"
