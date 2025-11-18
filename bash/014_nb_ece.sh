#!/bin/bash
#SBATCH --job-name=014_nb
#SBATCH --output=/home/george/codes/lepinet/slurm/slurmjob-%j.out
#SBATCH --error=/home/george/codes/lepinet/slurm/slurmjob-%j.err
#SBATCH --partition=GPU48
#SBATCH --ntasks=1                 # Number of tasks (processes)
#SBATCH --time=08:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --mail-type=FAIL           # Send email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=gmo@ecos.au.dk # Your email for notifications

# Pick a random port to avoid collisions
# PORT=$(shuf -i 8000-9999 -n 1)
PORT=8888

# Print node and port info for later use
echo "Jupyter running on node: $(hostname)"
echo "Using port: $PORT"

# Start reverse SSH tunnel in the background
ssh -o StrictHostKeyChecking=no -N -R ${PORT}:localhost:${PORT} gpucluster &

# Give SSH a few seconds to establish
sleep 5

# Start Jupyter Notebook or Lab
jupyter lab --no-browser --port=$PORT --ip=127.0.0.1