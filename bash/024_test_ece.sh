#!/bin/bash
#SBATCH --job-name=024_lepi_test_multihead
#SBATCH --output=/home/george/codes/lepinet/slurm/slurmjob-%j.out
#SBATCH --error=/home/george/codes/lepinet/slurm/slurmjob-%j.err
#SBATCH --partition=GPU24
#SBATCH --nodelist=node5
#SBATCH --ntasks=12                 # Number of tasks (processes)
#SBATCH --time=01:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --mail-type=FAIL            # Send email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=gmo@ecos.au.dk # Your email for notifications

# accelerate launch dev/022_lepi_large_prod_v3_multihead.py --config configs/20260211_train_ece.yaml
python dev/024_lepi_test_multihead.py --config configs/20260616_test_ece.yaml