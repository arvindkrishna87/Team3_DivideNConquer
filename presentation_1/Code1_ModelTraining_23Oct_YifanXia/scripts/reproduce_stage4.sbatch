#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-5%4
#SBATCH --job-name=Yifan_P1_repro_train
set -euo pipefail
: "${RUN_ROOT:?Set RUN_ROOT to the new prepared workspace}"
: "${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export MPLCONFIGDIR="$RUN_ROOT/analysis/training_tools/.mpl_cache/$SLURM_ARRAY_TASK_ID"
cd "$RUN_ROOT/code"
/projects/e32998/env/bin/python -u "$RUN_ROOT/analysis/training_tools/run_training.py" \
  --workspace "$RUN_ROOT" --split "$SLURM_ARRAY_TASK_ID"
