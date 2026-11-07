#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=uni2_embed
#SBATCH --output=precompute_uni2.out
#SBATCH --error=precompute_uni2.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jingyuhuo2028@u.northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

cd /home/cny2467/Presentation2_base_code

python3 precompute_pooled_features_uni2.py \
    --patches_dir /home/cny2467/patch_symlinks_odd \
    --embeddings_dir /home/cny2467/uni2_embeddings_1536
