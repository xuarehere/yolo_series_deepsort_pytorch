cd ..
python train.py   --img-size 640  --batch-size 32 --epochs 100 --data sucity-18-1_train_jiaoLian.yaml  --weights /workspace/models/yolov5/tensorrtx/yolov5s.pt    --device 0,1  --sync-bn
# python -m torch.distributed.launch --nproc_per_node 2  train.py   --img-size 640  --batch-size 128 --epochs 300 --data sucity-19-2_baoAn_merge.yaml  --weights /workspace/models/yolov5/tensorrtx/yolov5s.pt   --device 0,1 --sync-bn

