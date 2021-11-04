#!/bin/bash

cd `dirname "$BASH_SOURCE"`
cd ..
python main_STN_VAE.py \
--project "STN_VAE" \
--exp-name "deneme" \
--data_path /home/ikocdemi/workspace/DATASETS/ \
--batch_size 4 \
--download_dataset