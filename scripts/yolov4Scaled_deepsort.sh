###
 # @Author: xuarehere
 # @Date: 2022-09-01 21:22:49
 # @LastEditTime: 2022-09-01 22:47:55
 # @LastEditors: xuarehere
 # @Description: 
 # @FilePath: /yolovx_deepsort_pytorch/scripts/yolov4Scaled_deepsort.sh
 # 可以输入预定的版权声明、个性签名、空行等
### 
python deepsort.py ./001.avi --save_path ./output/yolov4Scaled/001 --detect_model yolov4Scaled --config_detection ./configs/yolov4Scaled.yaml 