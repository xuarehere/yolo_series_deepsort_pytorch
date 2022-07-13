'''
Descripttion: 
Author: Zhang Bingzhen
Date: 2021-07-28 17:37:27
LastEditors: Zhang Bingzhen
LastEditTime: 2021-07-29 14:51:25
FilePath: /deepsort_yolov5_tensorrt/pythontest/reidmodel.py
'''
import torch
from torch.autograd import Variable
# import torch.onnx as torch_onnx
import onnx

import torch
import torchvision.transforms as transforms
import numpy as np
# import cv2    
import logging

from original_model import Net

def main():
    # input_shape = (,3,128,64)
    batch_size = 1
    # model_onnx_path = "original_ckpt.onnx"
    # dummy_input = Variable(torch.randn(1, *input_shape))
    # model_path = '/workspace/py/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'
    model_path = '/workspace/models/deepsort/original_ckpt.t7'
    model_onnx_path = model_path.split('/')[-1].split(".")[0] + ".onnx"
    torch_model = Net(reid=True)
    device = "cuda"
    state_dict = torch.load(model_path, map_location=torch.device(device))['net_dict']
    # map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    input_x = torch.randn(batch_size, 3, 128, 64, requires_grad=True)
    torch_out = torch_model(input_x)
    torch.onnx.export(torch_model,               # model being run
                  input_x,                         # model input (or a tuple for multiple inputs)
                  model_onnx_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})
    print(torch_out)

def onnxcheck():
    model_onnx_path = "reidmodel.onnx"
    onnx_model = onnx.load(model_onnx_path)
    onnx.checker.check_model(onnx_model)

# class Extractor(object):
#     def __init__(self, model_path, use_cuda=True):
#         self.net = Net(reid=True)
#         self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
#         state_dict = torch.load(model_path, map_location=torch.device(self.device))[
#             'net_dict']
#         self.net.load_state_dict(state_dict)
#         logger = logging.getLogger("root.tracker")
#         logger.info("Loading weights from {}... Done!".format(model_path))
#         self.net.to(self.device)
#         self.size = (64, 128)
#         self.norm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ])

#     def _preprocess(self, im_crops):
#         """
#         TODO:
#             1. to float with scale from 0 to 1
#             2. resize to (64, 128) as Market1501 dataset did
#             3. concatenate to a numpy array
#             3. to torch Tensor
#             4. normalize
#         """
#         def _resize(im, size):
#             return cv2.resize(im.astype(np.float32)/255., size)

#         im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
#             0) for im in im_crops], dim=0).float()
#         return im_batch

#     def __call__(self, im_crops):
#         im_batch = self._preprocess(im_crops)
#         with torch.no_grad():
#             im_batch = im_batch.to(self.device)
#             features = self.net(im_batch)
#         return features.cpu().numpy()


if __name__ == '__main__':
    # img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    # extr = Extractor("checkpoint/ckpt.t7")
    # feature = extr(img)
    # print(feature.shape)
    main()
    # onnxcheck()