#!/bin/bash
#SBATCH -p pascal
#SBATCH --nodes 1
#SBATCH -J job_name

. /etc/profile.d/modules.sh
module load cuda
module load openmpi

python Comparison_DataNormalization.py
# mpiexec -np 4 --npernode 1 ./a.out
