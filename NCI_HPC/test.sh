#!/bin/bash

#PBS -l ncpus=12
#PBS -l mem=30GB
#PBS -l jobfs=200GB
#PBS -q gpuvolta
#PBS -P li96
#PBS -l walltime=10:00:00
#PBS -l storage=gdata/li96+scratch/li96
#PBS -l wd

module load python3/3.7.4
module load pytorch/1.9.0
cd /scratch/$PROJECT/$USER/progressive_fixing
pip install -r requirements.txt
python3 main.py > /scratch/$PROJECT/$USER/progressive_fixing/job_logs/$PBS_JOBID.log