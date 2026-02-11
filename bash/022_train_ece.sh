#!/bin/bash
#SBATCH --job-name=022_lepi_large_prod_v3_multihead
#SBATCH --output=/home/george/codes/lepinet/slurm/slurmjob-%j.out
#SBATCH --error=/home/george/codes/lepinet/slurm/slurmjob-%j.err
#SBATCH --partition=GPU24
#SBATCH --ntasks=12                 # Number of tasks (processes)
#SBATCH --time=10-08:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --mail-type=FAIL            # Send email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=gmo@ecos.au.dk # Your email for notifications

# accelerate launch dev/022_lepi_large_prod_v3_multihead.py --config configs/20260211_train_ece.yaml
python dev/022_lepi_large_prod_v3_multihead.py --config configs/20260211_train_ece.yaml