#!/bin/bash
cc hd_encoder.c --shared -fPIC -o hd_encoder_$(uname -i).so
cc hd_encoder_bv.c --shared -fPIC -o hd_encoder_bv_$(uname -i).so

