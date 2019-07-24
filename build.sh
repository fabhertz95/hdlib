#!/bin/bash
cc hd_encoder.c hd_classifier.c --shared -fPIC -o hdlib_$(uname -i).so

