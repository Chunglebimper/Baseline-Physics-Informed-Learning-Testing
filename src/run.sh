#!/bin/bash

# below are the baseline runs
#python main.py --use_glcm --batch_size 2 --patch_size 64 --stride 32 --epochs 31 --data_root ../data --oversample 10 

for i in 32 64
do
    for j in 4 8 16 
    do 
        python main.py --use_glcm --batch_size 2 --patch_size $i --stride $j --epochs 50 --data_root ../data --oversample 10 
    done
done


