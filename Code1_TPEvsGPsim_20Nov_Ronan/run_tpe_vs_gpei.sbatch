#!/bin/bash
# TPE vs GP-EI toy simulation, version 3: one array task per (setting, n_init) pair.
# 5 settings x 2 initial-trial counts = 10 tasks; %4 runs at most 4 at a time (4 CPUs in total).
# Each task takes about 5-25 minutes on one CPU. After all tasks finish, combine the results with:
#     python3 tpe_vs_gpei.py --merge results
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --array=0-4%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=tpe_vs_gpei
#SBATCH --output=tpe_vs_gpei_%A_%a.out
#SBATCH --error=tpe_vs_gpei_%A_%a.err

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

# One CPU per task: stop numpy/scipy from starting extra threads
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SETTINGS=(fixed noisy random fixed6d fixed10d)
INITS=(16)
SETTING=${SETTINGS[$((SLURM_ARRAY_TASK_ID % 5))]}
N_INIT=${INITS[$((SLURM_ARRAY_TASK_ID / 5))]}

cd "$SLURM_SUBMIT_DIR"
echo "Task $SLURM_ARRAY_TASK_ID: setting=$SETTING n_init=$N_INIT"
python3 -c "import optuna, sklearn; print('optuna', optuna.__version__, '| sklearn', sklearn.__version__)"
python3 tpe_vs_gpei.py --repeats 100 --settings "$SETTING" --n_init "$N_INIT" \
    --outdir "results/${SETTING}_init${N_INIT}"
