#!/bin/bash
#SBATCH --job-name=028_lepi_hierarchical_multihead_v4
#SBATCH --output=/home/george/codes/lepinet/slurm/slurmjob-%j.out
#SBATCH --error=/home/george/codes/lepinet/slurm/slurmjob-%j.err
#SBATCH --partition=GPU24
#SBATCH --nodelist=node5
#SBATCH --ntasks=12                 # Number of tasks (processes)
#SBATCH --time=10-08:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --mail-type=FAIL            # Send email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=gmo@ecos.au.dk # Your email for notifications

python dev/028_lepi_hierarchical_multihead_v4.py --config configs/20260707_train_multihead_v4_small.yaml
