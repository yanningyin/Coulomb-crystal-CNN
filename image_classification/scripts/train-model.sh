#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=slurm_logs/%j_%x.out  # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=slurm_logs/%j_%x.err  # where to store error messages
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

# Send some noteworthy information to the output log
SlurmID=$SLURM_JOBID
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

cd ..

# T
python3 -u cnn-one-label.py ./images_for_training/ -m "alexnet" -w -rgb -l t -rN "range(100, 300, 1)" -rT "range(5, 26, 2)"

# N
python3 -u cnn-one-label.py ./images_for_training/ -m "resnet18"  -w -l n -rN "range(100, 300, 1)" -rT "range(5, 41, 1)"


# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
