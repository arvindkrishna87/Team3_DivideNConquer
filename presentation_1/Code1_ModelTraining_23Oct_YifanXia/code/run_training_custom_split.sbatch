#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=Yifan_P1_s01
#SBATCH --chdir=/home/siw1333/Yifan_Xia_coder_p1/code
#SBATCH --output=/home/siw1333/Yifan_Xia_coder_p1/logs/train_split01_%j.out
#SBATCH --error=/home/siw1333/Yifan_Xia_coder_p1/logs/train_split01_%j.err

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export MPLCONFIGDIR=/home/siw1333/Yifan_Xia_coder_p1/analysis/training_tools/.mpl_cache

/projects/e32998/env/bin/python -u \
    /home/siw1333/Yifan_Xia_coder_p1/analysis/training_tools/run_training.py \
    --workspace /home/siw1333/Yifan_Xia_coder_p1 --split 1
