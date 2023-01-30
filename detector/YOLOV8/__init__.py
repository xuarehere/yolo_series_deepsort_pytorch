'''
Author: xuarehere xuarehere@foxmail.com
Date: 2023-01-16 16:32:14
LastEditTime: 2023-01-16 16:33:58
LastEditors: xuarehere xuarehere@foxmail.com
Description: 
FilePath: /yolovx_deepsort_pytorch/detector/YOLOV8/__init__.py

'''
import sys
sys.path.append("detector/YOLOV8")


# from .detector import hello
from .detector import YOLOv8
from . import  utils
__all__ = ['YOLOv8',
            'utils'
            ]



