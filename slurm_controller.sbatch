#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH --job-name=ipy-controller-
#SBATCH --ntasks=1
/home/ab5169/miniconda3/envs/linear-hvg/bin/python -m ipyparallel.controller --profile-dir="/home/ab5169/.ipython/profile_slurm" --cluster-id=""
