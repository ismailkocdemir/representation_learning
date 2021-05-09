#!/bin/bash

cd `dirname "$BASH_SOURCE"`
cd ..
python main_STN_VAE.py \
--data_path /HDD/DATASETS/ \
--batch_size 2
