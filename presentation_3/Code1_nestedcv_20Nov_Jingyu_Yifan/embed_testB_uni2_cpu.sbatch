#!/bin/bash
# UNI2 features from each patch at its original size. Each array task encodes every 4th patch.
# Rerunning skips patches already saved.
#SBATCH --account=e32998
#SBATCH --partition=normal
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=UNI2_NATIVE_embed
#SBATCH --output=/home/cny2467/Presentation3_base_code/logs/cpu_%x_%A_%a.out
#SBATCH --error=/home/cny2467/Presentation3_base_code/logs/cpu_%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jingyuhuo2028@u.northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p /home/cny2467/Presentation3_base_code/logs

cd /home/cny2467/Presentation3_base_code
python3 precompute_pooled_features.py \
    --encoder uni2 --uni2_native_size \
    --device cpu \
    --patches_dir /projects/e32998/patch_images \
    --embeddings_dir /home/cny2467/uni2_embeddings_1536_native_cpu \
    --batch_size 64 --num_workers 2 \
    --num_shards 4 --shard_index "$SLURM_ARRAY_TASK_ID" --log_every 20
