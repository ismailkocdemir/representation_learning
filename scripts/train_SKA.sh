#!/bin/bash

cd `dirname "$BASH_SOURCE"`
cd ..
python main_SKA.py \
--data-path /HDD/DATASETS/ \
--dataset-type 'Cifar100' \
--sim-loss \
--vico-mode 'vico_select' \
--no-hypernym \
--no-glove \