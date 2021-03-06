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

filename="force_47_171_172"
modelname="resnet50"
dataset="cifar10"
loss="cross_entropy"
PF_criterion="force"
PF_epoch_1=47
PF_epoch_2=171
PF_epoch_3=172
projectDIR="progressive_fixing"

module load python3/3.9.2
module load pytorch/1.9.0
cd /scratch/$PROJECT/$USER/$projectDIR
python3 train.py --dataset=$dataset --model=$modelname --loss=$loss --PF_criterion=$PF_criterion --PF_epoch_1=$PF_epoch_1 --PF_epoch_2=$PF_epoch_2 --PF_epoch_3=$PF_epoch_3 >/scratch/$PROJECT/$USER/progressive_fixing/NCI-HPC/logs/$filename.log
