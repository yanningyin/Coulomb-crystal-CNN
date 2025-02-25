#!/bin/bash
#SBATCH --job-name=generate_images
#SBATCH --output=slurm_logs/%j_%x.out  
#SBATCH --error=slurm_logs/%j_%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


# For using Slurm: save some noteworthy info in the output log
SlurmID=$SLURM_JOBID
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"


# Format to generate series of images using runMultiSim.sh
# ./runMultiSim.sh $1 $2 $3 $4 $5 $6, where
# $1: num_ions_start; $2: num_ions_end; $3: num_ions_increment
# $4: temperature_start; $5: temperature_end; $6: temperature_increment

./runMultiSim.sh 100 200 1 5 20 1

echo "Done."
