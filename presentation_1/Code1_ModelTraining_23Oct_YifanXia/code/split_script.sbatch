#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=Yifan_P1_splits
#SBATCH --chdir=/home/siw1333/Yifan_Xia_coder_p1/code
#SBATCH --output=/home/siw1333/Yifan_Xia_coder_p1/logs/splits_%j.out
#SBATCH --error=/home/siw1333/Yifan_Xia_coder_p1/logs/splits_%j.err

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

/projects/e32998/env/bin/python -u make_splits.py \
    --seed 3902026 \
    --save_dir /home/siw1333/Yifan_Xia_coder_p1/splits

/projects/e32998/env/bin/python -u /home/siw1333/Yifan_Xia_coder_p1/analysis/validate_splits.py \
    --workspace /home/siw1333/Yifan_Xia_coder_p1 \
    --seed 3902026 \
    --reference-splits /projects/e32998/STAT390_Fall2026/Team3_DivideNConquer/presentation_1/taylor_code/no_train_val_leakage_splits
