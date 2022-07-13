cd ..
python train.py   --img-size 640  --batch-size 256 --epochs 300 --data sucity-19-2_baoAn_default.yaml  --weights /workspace/py/ultralytics/yolov5_container/app/runs/train/exp12_baoAn_merge_v2.1/weights/best.pt   --device 0,1 
# python -m torch.distributed.launch --nproc_per_node 2  train.py   --img-size 640  --batch-size 128 --epochs 300 --data sucity-19-2_baoAn_merge.yaml  --weights /workspace/models/yolov5/tensorrtx/yolov5s.pt   --device 0,1 --sync-bn

