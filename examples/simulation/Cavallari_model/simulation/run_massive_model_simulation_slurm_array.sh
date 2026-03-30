#!/bin/bash
#SBATCH -p NOParalela
#SBATCH -J Cavallari_model_simulations
#SBATCH -t 24:00:00
#SBATCH --array=0-49
#SBATCH --cpus-per-task=56
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

SCRATCH_ROOT="/SCRATCH/TIC117/pablomc/Cavallari_model_simulations"

cd /LUSTRE/home/TIC117/pablomc/ncpi/examples/simulation/Cavallari_model/simulation || exit 1

source ~/.bashrc
conda activate ncpi-env

JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${SCRATCH_ROOT%/}"
exec >"${SCRATCH_ROOT%/}/slurm_${JOB_ID}_${TASK_ID}.out" \
     2>"${SCRATCH_ROOT%/}/slurm_${JOB_ID}_${TASK_ID}.err"

OUTPUT_ROOT="${SCRATCH_ROOT%/}/simulation_output"
mkdir -p "${OUTPUT_ROOT}"

python massive_model_simulation.py \
  --batch-id "${SLURM_ARRAY_TASK_ID}" \
  --num-batches "${SLURM_ARRAY_TASK_COUNT}" \
  --output-root "${OUTPUT_ROOT}"
