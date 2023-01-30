'''
Author: xuarehere xuarehere@foxmail.com
Date: 2023-01-16 14:57:17
LastEditTime: 2023-01-16 15:44:23
LastEditors: xuarehere xuarehere@foxmail.com
Description: 
FilePath: /yolov8/run.py

'''
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
# results = model.train(data="coco128.yaml", epochs=3)  # train the model
# results = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
results = model("./ultralytics/assets/bus.jpg")  # predict on an image
# success = model.export(format="onnx")  # export the model to ONNX format
print(111)