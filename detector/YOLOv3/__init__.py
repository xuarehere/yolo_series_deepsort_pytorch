import sys
sys.path.append("detector/YOLOv3")      # 工程运行的根目录是是跟 detector 是同级的


from .detector import YOLOv3    
__all__ = ['YOLOv3']



