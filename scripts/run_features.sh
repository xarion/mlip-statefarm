#!/bin/bash 
#SBATCH -p gpu 
#SBATCH -N 1 
#SBATCH -t 05:0:00 
module load cuda cudnn python/2.7.9
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hpc/sw/caffe-2015.11.30-gpu/lib
python features.py
echo "finished"
# sleep 300
