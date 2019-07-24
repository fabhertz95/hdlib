#!/bin/bash
cc hd_encoder.c --shared -fPIC -o hd_encoder_$(uname -i).so

