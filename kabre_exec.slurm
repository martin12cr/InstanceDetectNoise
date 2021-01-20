#!/bin/bash
#SBATCH --job-name=instance_selection_regENN_DiscENN
#SBATCH --output=regENN_DiscENN.txt
#SBATCH --partition=nu-wide
#SBATCH --ntasks=64
#SBATCH --time=24:00:00

srun ./modules/main.py