#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name=exp99
#SBATCH --array=0-29%10
#SBATCH --no-requeue
#SBATCH --output=/home/vnc0125/presentation_3/taylor_logs/%x_%A_%a.out
#SBATCH --error=/home/vnc0125/presentation_3/taylor_logs/%x_%A_%a.err

BASE=/home/vnc0125/presentation_3/base_code
SPLIT_DIR=$BASE/taylor_splits_99
EMB_DIR=/home/vnc0125/presentation_3/embeddings_uni2_99
SPLITS=(data_splits_new_01.npz data_splits_new_02.npz data_splits_new_03.npz data_splits_new_04.npz data_splits_new_05.npz)
TAGS=(baseline stain0.3 slice0.3 emb64 emb128 emb256)
FLAGS=("" "--stain_dropout 0.3" "--slice_dropout 0.3" "--embed_dim 64" "--embed_dim 128" "--embed_dim 256")

IDX=$SLURM_ARRAY_TASK_ID
SPLIT=${SPLITS[$(( IDX % 5 ))]}
CFG=$(( IDX / 5 ))

# every task gets its OWN folder (job id + task number), so runs can never share or overwrite one
export RUNS_DIR=/home/vnc0125/presentation_3/taylor_runs/job_${SLURM_ARRAY_JOB_ID}/task_${IDX}
if [ -e "$RUNS_DIR" ]; then
    echo "ERROR: $RUNS_DIR already exists; refusing to overwrite"
    exit 2
fi

# refuse to start if an input is missing
[ -f "$SPLIT_DIR/$SPLIT" ] || { echo "ERROR: missing split file $SPLIT_DIR/$SPLIT"; exit 3; }
[ -d "$EMB_DIR" ] || { echo "ERROR: missing embedding folder $EMB_DIR"; exit 3; }
mkdir -p "$RUNS_DIR"

sleep $(( IDX * 3 ))

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

echo "Task $IDX | tag=${TAGS[$CFG]} | split=$SPLIT | flags=${FLAGS[$CFG]}"
echo "Run folder parent: $RUNS_DIR"
cd $BASE
python3 main.py \
    --labels_csv $BASE/extra_benign_case_grade_match.csv \
    --patches_dir /projects/e32998/patch_images \
    --emb_dir $EMB_DIR \
    --load_splits $SPLIT_DIR/$SPLIT \
    --seed 42 --tag ${TAGS[$CFG]} \
    --per_slice_cap 500 --max_slices_per_stain 5 --clean_train_loss \
    ${FLAGS[$CFG]} && touch "$RUNS_DIR/FINISHED"
