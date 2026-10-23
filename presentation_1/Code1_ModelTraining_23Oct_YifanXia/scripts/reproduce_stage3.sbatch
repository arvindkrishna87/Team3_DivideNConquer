#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=Yifan_P1_repro_splits
set -euo pipefail
: "${RUN_ROOT:?Set RUN_ROOT to the new prepared workspace}"
: "${REFERENCE_SPLITS:?Set REFERENCE_SPLITS to the five Taylor reference split files}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
cd "$RUN_ROOT/code"
/projects/e32998/env/bin/python -u make_splits.py --seed 3902026 --save_dir "$RUN_ROOT/splits"
/projects/e32998/env/bin/python -u "$RUN_ROOT/analysis/validate_splits.py" \
  --workspace "$RUN_ROOT" --seed 3902026 --reference-splits "$REFERENCE_SPLITS"
