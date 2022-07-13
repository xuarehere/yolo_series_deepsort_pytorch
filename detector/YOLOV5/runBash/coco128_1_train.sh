cd ..
# CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights /workspace/models/yolov5/tensorrtx/yolov5s.pt
CUDA_VISIBLE_DEVICES=0,1 python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights /workspace/models/yolov5/tensorrtx/yolov5s.pt
