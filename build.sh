#!/bin/bash
cc hd_encoder.c hd_classifier.c --std=gnu99 -O3 --shared -fPIC -o hdlib_$(uname -m).so
cc hd_encoder.c hd_classifier.c --std=gnu99 test_inference.c -O3 -o test_inference
