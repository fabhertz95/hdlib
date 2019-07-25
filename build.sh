#!/bin/bash
cc hd_encoder.c hd_classifier.c -O3 --shared -fPIC -o hdlib_$(uname -m).so
cc hd_encoder.c hd_classifier.c test_inference.c -O3 -o test_inference
