#!/bin/bash

#PBS -l ncpus=12
#PBS -l mem=30GB
#PBS -l jobfs=200GB
#PBS -l ngpus=1
#PBS -q gpuvolta
#PBS -P li96
#PBS -l walltime=10:00:00
#PBS -l storage=gdata/li96+scratch/li96
#PBS -l wd

filename="UP2"
modelname="resnet50"
dataset="cifar10"
loss="cross_entropy"
PF_criterion="UP"
alpha=2
projectDIR="progressive_fixing"

module load python3/3.9.2
module load pytorch/1.9.0
cd /scratch/$PROJECT/$USER/$projectDIR
python3 train.py --dataset=$dataset --model=$modelname --loss=$loss --PF_criterion=$PF_criterion --UP_alpha=$alpha >/scratch/$PROJECT/$USER/progressive_fixing/NCI-HPC/logs/$filename.log
