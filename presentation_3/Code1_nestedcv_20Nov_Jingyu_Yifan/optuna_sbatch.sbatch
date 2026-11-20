#!/bin/bash
# One array task per outer test group. Each task tunes on the other four groups,
# refits, and evaluates the held-out group once.
#SBATCH --account=e32998
#SBATCH --partition=long
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH --job-name=NESTED_CV
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your-email@northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Start time: $(date)"
echo "Host: $(hostname)"

mkdir -p logs

cd /path/to/Presentation3_base_code

FOLDS=(A B C D E)
OUTER=${FOLDS[$SLURM_ARRAY_TASK_ID]}

python3 nested_cv.py \
    --outer_fold "$OUTER" \
    --groups_npz nested_splits/nested_groups.npz \
    --embeddings_dir /path/to/uni2_embeddings \
    --labels_csv /path/to/labels.csv \
    --patches_dir /projects/e32998/patch_images \
    --output_dir nested_cv_runs \
    --n_trials 40 \
    --sobol_startup_trials 16 \
    --epochs 30 \
    --batch_size 1 \
    --num_workers 2 \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --seed 42

echo "End time: $(date)"
