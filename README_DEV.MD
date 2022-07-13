

# YOLOV6
```
python deepsort.py ./001.avi --save_path ./output/yolov6/001 --detect_model yolov6 --config_detection ./configs/yolov6.yaml
```


# Q&A
## yolov4 No module named mish_cuda
No module named mish_cuda的問題，解決方法可參考這裡進行安裝，步驟7需要將檔案複製到site_package所在資料夾。可以透過以下指令顯示路徑位置：
python -c 'import site; print(site.getsitepackages())'
['/opt/conda/lib/python3.6/site-packages']
或是使用
pip install git+https://github.com/thomasbrandon/mish-cuda/安裝


to do list
- [ ] 保存的视频流卡顿。根据给定的视频流计算推理的 fps，之后设定视频保存的 fps 参数
- [x] 单目标跟做，多个目标，同时跟踪呢？
> mask = cls_ids 筛选出需要的目标，写成配置形式
- [x] 检测的 bbox 与原始框的关系，即 图片 bbox 是通过 resize 后的图片得到的，是怎么跟原来的图片的位置对应上的
> 计算位置比例，之后进行计算就可以了
- [ ] 生成 results.txt 结果上，每一次都是重复写入，即第 n 帧的结果写入之后，到 n+ m 帧时候，先擦除 n 帧的结果，再重新写入 n 帧的结果，之后再写 m 帧的结果。