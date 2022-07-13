import sys
sys.path.append("detector/YOLOV6")


from .detector import YOLOv6
from . import tools
__all__ = ['YOLOv6',
            'tools',
            ]



