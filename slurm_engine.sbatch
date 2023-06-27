#!/bin/sh
#SBATCH --ntasks=255
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=ipy-engine-
srun ipengine --profile-dir="/home/ab5169/.ipython/profile_slurm" --cluster-id=""
