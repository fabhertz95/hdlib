#!/bin/bash
cc test.c --shared -fPIC -o test_$(uname -i).so
cc hd_encoder_bv.c --shared -fPIC -o hd_encoder_bv_$(uname -i).so

