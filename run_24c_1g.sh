#!/bin/bash -l

#SBATCH --job-name="train"    # Job name
#SBATCH --nodes=1             # Total number of nodes requested
#SBATCH --ntasks=1           # Total number of mpi tasks
#SBATCH --cpus-per-task=2     # Total number of CPUs per task
###SBATCH --ntasks-per-node=8  # Number of tasks per each node
###SBATCH --ntasks-per-core=1  # Number of tasks per each core
###SBATCH --mem-per-cpu=4800    # Memory (in MB)
###SBATCH --time=7-0:00:00    # Run time (hh:mm:ss)
#SBATCH --partition=chalawan_gpu       # Name of gpu partition
###SBATCH --nodelist=pollux[2-3] 
#SBATCH --gres=gpu:1          # Name of the generic consumable resource

cd ${SLURM_SUBMIT_DIR}
PYTHON=/lustre/rangsiman/miniconda3/envs/tf-gpu/bin/python 
$PYTHON train.py > train.out
 
