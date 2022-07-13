#!/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=4,5,6,7  python train.py --img 640 --batch 32  --epochs 100 --data data/sucity-19.yaml --weights /workspace/models/yolov5/tensorrtx/yolov5s.pt
