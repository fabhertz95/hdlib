#!/bin/bash
#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli

#Titan X: compute_52, TX1: compute_53, GTX1080Ti: compute_61, TX2: compute_62
nvccflags="-O3 --use_fast_math -std=c++11 -Xcompiler '-fopenmp' --shared --gpu-architecture=compute_53 --compiler-options -fPIC --linker-options --no-undefined"
nvcc hd_encoder.c hd_classifier.c test_inference.c -o test_inference_$(uname -m) $nvccflags
nvcc hd_encoder.c hd_classifier.c -o hdlib_$(uname -m) $nvccflags
