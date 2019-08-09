#!/bin/bash
cc hd_encoder.c hd_batch_encoder.c hd_classifier.c --std=c11 -O3 --shared -fPIC -o hdlib_$(uname -m).so
cc hd_encoder.c hd_batch_encoder.c hd_classifier.c --std=c11 test_inference.c -O3 -o test_inference
