#!/bin/bash

#SBATCH -p barbun-cuda 
#SBATCH -A ikocdemir
#SBATCH -J ska                          
#SBATCH --workdir=/truba/home/ikocdemir/representation_learning/
#SBATCH --nodes=1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2                                               
#SBATCH --output=/truba/home/ikocdemir/representation_learning/slurm_out/slurm-%j.out
#SBATCH --error=/truba/home/ikocdemir/representation_learning/slurm_out/slurm-%j.err
#SBATCH --time=1-00:00:00				                            
#SBATCH --no-requeue 

source ~/.bashrc
conda activate h3dr

module load centos7.3/comp/gcc/7
module load centos7.3/comp/cmake/3.18.0
module load centos7.3/lib/cuda/10.1

cd /truba/home/ikocdemir/representation_learning/

python main_SKA.py \
--data-path /truba/home/ikocdemir/data/ \
--embed-path /truba/home/ikocdemir/data/pretrained_embeddings \
--download-dataset \
--dataset-type 'Cifar100' \
--batch-size 512 \
--lr 1e-1 \
--num-epochs 700 \
--warmup-epochs 10 \
--sim-loss \
--vico-mode 'vico_select' \
--no-hypernym \
--no-glove \

echo "JOB DONE!!!"

