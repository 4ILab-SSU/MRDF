#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=2 python train_simple.py --data_root data/FakeAVCeleb_v1.2 --model_type MRDF_CE --batch_size 16 --max_epochs 30 --learning_rate 1e-3 --gpus 1 --outputs ./outputs