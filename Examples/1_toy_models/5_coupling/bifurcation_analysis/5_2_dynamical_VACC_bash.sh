#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=40G
#SBATCH --account=pi-dbernste
#SBATCH --job-name=chaos
#SBATCH --array=1-25
#SBATCH --output=logs/trash/%x_%A_%a.out
#SBATCH --error=logs/trash/%x_%A_%a.err
#SBATCH --time=120:00

source ~/.bashrc
set -x

conda activate M3

array_task_idx="${SLURM_ARRAY_TASK_ID}"
array_total_runs=100
bifurc_points_total=500
python3.10 chaos_script.py "${array_task_idx}" "${array_total_runs}" "${bifurc_points_total}"