#!/bin/bash
#SBATCH --job-name=017_integration_asgers_heads
#SBATCH --output=/home/george/codes/lepinet/slurm/slurmjob-%j.out
#SBATCH --error=/home/george/codes/lepinet/slurm/slurmjob-%j.err
#SBATCH --partition=GPU48
#SBATCH --ntasks=1                 # Number of tasks (processes)
#SBATCH --time=08:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --mail-type=ALL            # Send email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=gmo@ecos.au.dk # Your email for notifications

python dev/017_integration_asgers_heads.py --config configs/20251124_train_ece.yaml
# py-spy record --output /home/george/codes/lepinet/slurm/profile.pyspy --format raw --duration 120 --full-filenames -- python dev/011_lepi_large_prod_v2.py --config configs/20251111_train_ece.yaml