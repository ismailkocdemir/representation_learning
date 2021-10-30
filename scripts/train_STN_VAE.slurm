#!/bin/bash

#SBATCH -p palamut-cuda 
#SBATCH -A romer
#SBATCH -c 16
#SBATCH -n 1
#SBATCH -J stn-vae
#SBATCH --workdir=/truba/home/ikocdemir/representation_learning/               
#SBATCH --gres=gpu:1                                               
#SBATCH --output=/truba/home/ikocdemir/representation_learning/slurm_out/slurm-%j.out
#SBATCH --error=/truba/home/ikocdemir/representation_learning/slurm_out/slurm-%j.err
#SBATCH --time=1-00:00:00				                            
#SBATCH --no-requeue 

source ~/.bashrc
conda activate romer-gpu

cd /truba/home/ikocdemir/representation_learning/

python main_STN_VAE.py \
--project "STN_VAE" \ 
--exp-name "deneme" \
--data_path /truba/home/ikocdemir/data \
--batch_size 8 \
--download_dataset

echo "JOB DONE!!!"

