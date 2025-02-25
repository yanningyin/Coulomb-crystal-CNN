#!/bin/bash

cd ..

# T
python3 cnn-one-label.py ../images/images_to_eval \
        -md test \
        -mp output/t_5_25_2_alexnet_model_20250130204033.pth\
        -rgb \
        -l t

# N
python3 cnn-one-label.py ../images/images_to_eval\
        -md test \
        -mp output/n_100_299_1_resnet18_model_20240803110259.pth\
        -l n      