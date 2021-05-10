#!/bin/bash

#SBATCH -p barbun-cuda 
#SBATCH -A ikocdemir
#SBATCH -J cgid                          
#SBATCH --workdir=/truba/home/ikocdemir/tmo_detector
#SBATCH --nodes=1
#SBATCH --cpus-per-task 10
#SBATCH -ntasks-per-node=2
#SBATCH --gpus=2                                               
#SBATCH --output=/truba/home/ikocdemir/tmo_detector/slurm_out/cuda-slurm-%j.out
#SBATCH --error=/truba/home/ikocdemir/tmo_detector/slurm_out/cuda-slurm-%j.err
#SBATCH --time=0-13:00:00				                            
#SBATCH --no-requeue 


source ~/.bashrc
conda activate h3dr

module load centos7.3/comp/gcc/7
module load centos7.3/comp/cmake/3.18.0
module load centos7.3/lib/cuda/10.1

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000

python main_SKA.py \
--dist_url $dist_url
--data-path /truba/home/ikocdemir/data/ \
--embed-path /truba/home/ikocdemir/data/pretrained_embeddings \
--download-dataset \
--dataset-type 'Cifar100' \
--sim-loss \
--vico-mode 'vico_select' \
--no-hypernym \
--no-glove \

echo "JOB DONE!!!"

