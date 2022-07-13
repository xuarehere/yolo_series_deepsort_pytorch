cd ..
python train.py   --img-size 640  --batch-size 64 --epochs 80 --data sucity-18-2_train_baoan_jiaolian_add_futian_special.yaml  --weights /workspace/models/yolov5/tensorrtx/yolov5s.pt    --device 0,1  --sync-bn  --epochs_save 5
# python -m torch.distributed.launch --nproc_per_node 2  train.py   --img-size 640  --batch-size 128 --epochs 300 --data sucity-19-2_baoAn_merge.yaml  --weights /workspace/models/yolov5/tensorrtx/yolov5s.pt   --device 0,1 --sync-bn

