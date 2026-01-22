#!/bin/bash

# LOL-v1
python3 Enhancement/test_from_dataset.py --opt Options/LOL_v1.yml --weights pretrained_weights/LOL_v1.pth --dataset LOL_v1

# LOL-v2-real
python3 Enhancement/test_from_dataset.py --opt Options/LOL_v2_real.yml --weights pretrained_weights/LOL_v2_real.pth --dataset LOL_v2_real

# LOL-v2-synthetic
python3 Enhancement/test_from_dataset.py --opt Options/LOL_v2_synthetic.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset LOL_v2_synthetic

# SID
python3 Enhancement/test_from_dataset.py --opt Options/SID.yml --weights pretrained_weights/SID.pth --dataset SID

# SMID
python3 Enhancement/test_from_dataset.py --opt Options/SMID.yml --weights pretrained_weights/SMID.pth --dataset SMID

# SDSD-indoor
python3 Enhancement/test_from_dataset.py --opt Options/SDSD_indoor.yml --weights pretrained_weights/SDSD_indoor.pth --dataset SDSD_indoor

# SDSD-outdoor
python3 Enhancement/test_from_dataset.py --opt Options/SDSD_outdoor.yml --weights pretrained_weights/SDSD_outdoor.pth --dataset SDSD_outdoor