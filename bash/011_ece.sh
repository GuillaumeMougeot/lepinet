#!/bin/bash
#SBATCH --job-name=011_train_insect_classifier
#SBATCH --output=/home/george/codes/lepinet/slurm/slurmjob-%j.out
#SBATCH --error=/home/george/codes/lepinet/slurm/slurmjob-%j.err
#SBATCH --partition=GPU48
#SBATCH --ntasks=1                 # Number of tasks (processes)
#SBATCH --time=00:10:00            # Maximum runtime (HH:MM:SS)
#SBATCH --mail-type=ALL            # Send email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=gmo@ecos.au.dk # Your email for notifications

python dev/011_lepi_large_prod_v2.py --config configs/20251103_1.yaml