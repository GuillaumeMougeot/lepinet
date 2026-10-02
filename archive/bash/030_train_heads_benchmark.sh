#!/bin/bash
#SBATCH --job-name=030_hierarchical_heads_benchmark
#SBATCH --output=/home/george/codes/lepinet/slurm/slurmjob-%j.out
#SBATCH --error=/home/george/codes/lepinet/slurm/slurmjob-%j.err
#SBATCH --partition=GPU24
#SBATCH --nodelist=node5
#SBATCH --ntasks=12                 # Number of tasks (processes)
#SBATCH --time=10-08:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --mail-type=FAIL            # Send email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=gmo@ecos.au.dk # Your email for notifications

# One job per head type -- run all three to compare.
python dev/030_hierarchical_heads_benchmark.py --config configs/20260707_train_heads_benchmark_small_hierarchical.yaml
python dev/030_hierarchical_heads_benchmark.py --config configs/20260707_train_heads_benchmark_small_independent.yaml
python dev/030_hierarchical_heads_benchmark.py --config configs/20260707_train_heads_benchmark_small_autoregressive.yaml
