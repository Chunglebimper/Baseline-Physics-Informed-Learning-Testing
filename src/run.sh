#!/bin/bash

# below are the baseline runs
#python main.py --use_glcm --batch_size 2 --patch_size 64 --stride 32 --epochs 31 --data_root ../data --oversample 10 

source .venv/bin/activate

for i in 64 32
do
    for j in 16 8 4
    do 
        python main.py --use_glcm --batch_size 2 --patch_size $i --stride $j --epochs 50 --data_root ../data --oversample 10 --save_name test
    done
done


