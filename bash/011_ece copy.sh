#!/bin/bash
#SBATCH -C balder
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1          
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=48
#SBATCH --output /home/george/codes/lepinet/slurm/slurmjob-%j.out
#SBATCH --error /home/george/codes/lepinet/slurm/slurmjob-%j.err
#SBATCH --job-name=011_train_insect_classifier
#SBATCH --mail-user=gmo@ecos.au.dk
#SBATCH --mail-type=ALL

python dev/011_lepi_large_prod_v2.py --config configs/20251103_1.yaml