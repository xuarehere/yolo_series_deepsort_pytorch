import json
import torch
import torch.backends.cudnn as cudnn
import torchvision

import argparse
import os
from model import Net
from original_model import Net as original_Net
import evaluate  as evaluate

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir",default='data',type=str)
parser.add_argument("--no-cuda",action="store_true")
parser.add_argument("--gpu-id",default=0,type=int)
args = parser.parse_args()

# device
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# data loader
root = args.data_dir
query_dir = os.path.join(root,"query")
gallery_dir = os.path.join(root,"gallery")
# gallery_dir = os.path.join(root,"bounding_box_test")
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128,64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
queryloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(query_dir, transform=transform),
    batch_size=1000, shuffle=False
)
galleryloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(gallery_dir, transform=transform),
    batch_size=1000, shuffle=False
)

# model_path = "/workspace/py/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"          # Acc top1:0.985
                                                                                            # 输出 [batch_size, 512]
# model_path = "/workspace/py/deep_sort_pytorch/deep_sort/deep/checkpoint/original_ckpt.t7"   # Acc top1:0.970
                                                                                            # 输出 [batch_size, 128]
# model_path = "/workspace/py/deep_sort_pytorch/checkpoint/original38.t7"                                                                                        
model_path = "/workspace/py/deep_sort_pytorch/checkpoint/original_e75acc92.54327563249001.t7"
# net definition
print(os.getcwd())      # /workspace/py/deep_sort_pytorch
assert os.path.isfile(model_path), "Error: no checkpoint file found!"
print('Loading from :', model_path)

if "original" in model_path:
    net = original_Net(reid=True) 
    checkpoint = torch.load(model_path, map_location=torch.device(device))
else:
    net = Net(reid=True)
    checkpoint = torch.load(model_path)


net = Net(reid=True)
checkpoint = torch.load(model_path)

net_dict = checkpoint['net_dict']
net.load_state_dict(net_dict)
# net.load_state_dict(net_dict, strict=False)
net.eval()
net.to(device)

# compute features
query_features = torch.tensor([]).float()
query_labels = torch.tensor([]).long()
gallery_features = torch.tensor([]).float()
gallery_labels = torch.tensor([]).long()



with torch.no_grad():
    for idx,(inputs,labels) in enumerate(queryloader):
        inputs_numpy = inputs.numpy()       # inputs:torch.Size([1000, 3, 128, 64])
        # if idx % 100 == 0:
        print("queryloader index:{}, input len:{}".format(idx, len(inputs)))
        inputs = inputs.to(device)
        features = net(inputs).cpu()        # torch.Size([1000, 512]), 
        query_features = torch.cat((query_features, features), dim=0)       # torch.Size([1000, 512])
        query_labels = torch.cat((query_labels, labels))


    for idx,(inputs,labels) in enumerate(galleryloader):        # [64, 3, 128, 64], torch.Size([64])
        
        print("galleryloader index:{}, input len:{}".format(idx, len(inputs)))
        inputs = inputs.to(device)
        features = net(inputs).cpu()                # [batch, out_dim]      
        gallery_features = torch.cat((gallery_features, features), dim=0)
        gallery_labels = torch.cat((gallery_labels, labels))

gallery_labels -= 2     # torch.Size([19732])       # 为啥减去2


file_name = "output_q_concat_server.json"
ret_q = {}
ret_q["output_q_concat"] = query_features.tolist()
with open(file_name,'w') as file_object:
    json.dump(ret_q,file_object)

file_name = "labels_q_concat_server.json"
ret_q = {}
ret_q["labels_q_concat"] = query_labels.tolist()
with open(file_name,'w') as file_object:
    json.dump(ret_q,file_object)


file_name = "output_g_concat_server.json"
ret_g = {}
ret_g["output_g_concat"] = gallery_features.tolist()
with open(file_name,'w') as file_object:
    json.dump(ret_g,file_object)

file_name = "labels_g_concat_server.json"
ret_q = {}
ret_q["labels_g_concat"] = gallery_labels.tolist()
with open(file_name,'w') as file_object:
    json.dump(ret_q,file_object)


# save features
features = {
    "qf": query_features,
    "ql": query_labels,
    "gf": gallery_features,
    "gl": gallery_labels
}
evaluate.evaluate(features)

torch.save(features,"features_1.pth")

