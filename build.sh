#!/bin/bash
cc test.c --shared -fPIC -o test_$(uname -i).so

