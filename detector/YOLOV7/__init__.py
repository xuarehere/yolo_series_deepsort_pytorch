import sys
sys.path.append("detector/YOLOV7")


# from .detector import hello
from .detector import YOLOv7
from . import models, utils
__all__ = ['YOLOv7',
            'models',
            'utils'
            ]



