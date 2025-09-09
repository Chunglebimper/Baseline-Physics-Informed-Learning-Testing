#!/bin/bash

# below are the baseline runs
python main.py --use_glcm --batch_size 2 --patch_size 64 --stride 32 --epochs 31 --data_root ../data --oversample 10 
