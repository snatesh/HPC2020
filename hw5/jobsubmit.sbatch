#!/bin/bash

#SBATCH --job-name=jacobi2D
#SBATCH --nodes=8
#SBATCH --tasks-per-node=8
#SBATCH --time=00:02:00
#SBATCH --mem=1GB
#SBATCH --output=64tasks.out

/bin/hostname
/bin/pwd 
cd /home/srn324/hw5
module load openmpi/gnu/4.0.2
mpirun ./ssort
