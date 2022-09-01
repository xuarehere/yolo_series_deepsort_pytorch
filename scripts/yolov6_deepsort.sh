###
 # @Author: xuarehere
 # @Date: 2022-09-01 21:05:12
 # @LastEditTime: 2022-09-01 21:18:17
 # @LastEditors: xuarehere
 # @Description: 
 # @FilePath: /yolovx_deepsort_pytorch/scripts/yolov6_deepsort.sh
 # 可以输入预定的版权声明、个性签名、空行等
### 
python deepsort.py ./001.avi --save_path ./output/yolov6/001 --detect_model yolov6 --config_detection ./configs/yolov6.yaml 