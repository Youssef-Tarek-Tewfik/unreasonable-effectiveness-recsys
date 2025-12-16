#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH --output=./output/%x-%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=youssef.abdou@student.uni-siegen.de

#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --error=./output/%x-%j.err


# Modules
module load GpuModules
module load ohpc-compat/2.0
module load singularity/3.7.1


# Environment Variables (Note: for LensKit set non LK_* variables to 1 otherwise NCPUs)
export LK_NUM_PROCS=32
export LK_NUM_THREADS=2

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export NUMBA_NUM_THREADS=64


# Run
singularity exec --nv --pwd /mnt --bind ./:/mnt ./singularity/container.sif python -u -m source.run