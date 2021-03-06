#!/bin/bash

cd `dirname "$BASH_SOURCE"`
cd ..
python main_SKA.py \
--data-path /HDD/DATASETS/ \
--embed-path /HDD/DATASETS/pretrained_embeddings \
--dataset-type 'Cifar100' \
--sim-loss \
--vico-mode 'vico_select' \
--no-hypernym \
--no-glove \
--dump-path ./experiments/SKA/default