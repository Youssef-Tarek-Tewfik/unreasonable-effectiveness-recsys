#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --output=./output/%x-%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=youssef.abdou@student.uni-siegen.de

#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --error=./output/%x-%j.err


module load GpuModules
module load ohpc-compat/2.0
module load singularity/3.7.1


export LK_NUM_PROCS=8
export LK_NUM_THREADS=8
export LK_NUM_BACKEND_THREADS=2
export LK_NUM_CHILD_THREADS=2

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2


singularity exec --nv --pwd /mnt --bind ./:/mnt ./singularity/container.sif python -u -m source.run