#!/bin/bash
#SBATCH -p NOParalela
#SBATCH -J LIF_model_simulations
#SBATCH -t 96:00:00
#SBATCH --array=0-79
#SBATCH --cpus-per-task=56
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

SCRATCH_ROOT="/SCRATCH/TIC117/pablomc/LIF_model_simulations"

# --- move to your project directory ---
cd /LUSTRE/home/TIC117/pablomc/ncpi/examples/simulation/Hagen_model/simulation || exit 1

# --- activate environment ---
source ~/.bashrc
conda activate ncpi-env

# --- set up paths and environment variables ---
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${SCRATCH_ROOT%/}" # ensure scratch directory exists
exec >"${SCRATCH_ROOT%/}/slurm_${JOB_ID}_${TASK_ID}.out" \
     2>"${SCRATCH_ROOT%/}/slurm_${JOB_ID}_${TASK_ID}.err"

# --- create output directory for this job ---
OUTPUT_ROOT="${SCRATCH_ROOT%/}/simulation_output"
mkdir -p "${OUTPUT_ROOT}"

# --- run your script ---
python massive_model_simulation.py \
  --batch-id "${SLURM_ARRAY_TASK_ID}" \
  --num-batches "${SLURM_ARRAY_TASK_COUNT}" \
  --output-root "${OUTPUT_ROOT}"
