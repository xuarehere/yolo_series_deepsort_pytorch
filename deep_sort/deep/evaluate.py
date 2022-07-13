import torch

# model_path = "/workspace/py/deep_sort_pytorch/deep_sort/deep/checkpoint/features.pth"
# features = torch.load(model_path)
# qf = features["qf"]
# ql = features["ql"]
# gf = features["gf"]
# gl = features["gl"]

# scores = qf.mm(gf.t())
# res = scores.topk(5, dim=1)[1][:,0]
# top1correct = gl[res].eq(ql).sum().item()

# print("Acc top1:{:.3f}".format(top1correct/ql.size(0)))


def evaluate(features):
    qf = features["qf"]     # torch.Size([3368, 512])
    ql = features["ql"]
    gf = features["gf"]     # torch.Size([19732, 512])
    gl = features["gl"]
    # 1. gf 进行转置；2. qf 与 gf 转置结果进行相乘,作用是什么？为什么可以求得分
    scores = qf.mm(gf.t())  # gf.t() 转置，gf.shape=[19732, 128]  =>  gf.t().shape = [128, 19732]
    # 取出 top1 的结果
    # 1. scores.topk(5, dim=1)  对第一个维度求 top 5 的结果；2.scores.topk(5, dim=1)[1] 取第一个维度的结果，就是label id 信息；3. ...[:,0] 取出第一维度信息，也就是 top1
    res = scores.topk(5, dim=1)[1][:,0]
    # 算 top1 的结果与底库的准确度，算top1 的准确度
    top1correct = gl[res].eq(ql).sum().item()

    print("Acc top1:{:.3f}".format(top1correct/ql.size(0))) # Acc top1:0.985
    pass


def load_json_data_and_to_torch(path_dir):
    import os 
    import json
    import numpy as np
    features_json = None
    def load_json(path):
        with open(path,'r') as file_object:
            ret =  json.load(file_object)
        return ret

    # nx_output_g_concat_path = "/workspace/py/deep_sort_pytorch/results_analysis/nx_reid_result/output_g_concat.json"
    # # with open(nx_output_g_concat_path,'r') as file_object:
    # #     nx_output_g_concat = json.load(file_object)
    # nx_output_g_concat = load_json(nx_output_g_concat_path)
    # nx_output_g_concat_list = np.array( 
    #     nx_output_g_concat['output_g_concat'] 
    # )      

    # output_q_concat = load_json(os.path.join(path_dir, "output_q_concat.json"))["output_q_concat"]
    # labels_q_concat = load_json(os.path.join(path_dir, "labels_q_concat.json"))["labels_q_concat"]
    # output_g_concat = load_json(os.path.join(path_dir, "output_g_concat.json"))["output_g_concat"]
    # labels_g_concat = load_json(os.path.join(path_dir, "labels_g_concat.json"))["labels_g_concat"]

    output_q_concat = torch.from_numpy( np.array(load_json(os.path.join(path_dir, "output_q_concat.json"))["output_q_concat"]))
    labels_q_concat = torch.from_numpy( np.array(load_json(os.path.join(path_dir, "labels_q_concat.json"))["labels_q_concat"]))
    output_g_concat = torch.from_numpy( np.array(load_json(os.path.join(path_dir, "output_g_concat.json"))["output_g_concat"]))
    labels_g_concat = torch.from_numpy( np.array(load_json(os.path.join(path_dir, "labels_g_concat.json"))["labels_g_concat"]))	

    if output_q_concat.dtype == torch.float64:
        output_q_concat = torch.tensor(output_q_concat, dtype=torch.float32)
        labels_q_concat = torch.tensor(labels_q_concat, dtype=torch.float32)
        output_g_concat = torch.tensor(output_g_concat, dtype=torch.float32)
        labels_g_concat = torch.tensor(labels_g_concat, dtype=torch.float32)
    # a = load_json(os.path.join(path_dir, "output_g_concat.json")["output_g_concat"])
	# output_q_concat  = torch.from_numpy( np.array(load_json(os.path.join(path_dir, "output_q_concat.json"))))
	# labels_q_concat  = torch.from_numpy( np.array(	load_json(os.path.join(path_dir, "labels_q_concat.json"))))
    # output_g_concat  = torch.from_numpy( np.array(	load_json(os.path.join(path_dir, "output_g_concat.json")["output_g_concat"])))
    # labels_g_concat  = torch.from_numpy( np.array(	load_json(os.path.join(path_dir, "labels_g_concat.json")["labels_g_concat"])))

    features_json = {
        "qf": output_q_concat,
        "ql": labels_q_concat,
        "gf": output_g_concat,
        "gl": labels_g_concat
    }
    return features_json

def evaluate_two(features, features_nx):
    qf = features["qf"]     # torch.Size([3368, 512])
    ql = features["ql"]
    gf = features["gf"]     # torch.Size([19732, 512])
    gl = features["gl"]

       
    qf_nx = features_nx["qf"]     # torch.Size([3368, 512])
    ql_nx = features_nx["ql"]
    gf_nx = features_nx["gf"]     # torch.Size([19732, 512])
    gl_nx = features_nx["gl"]

    scores = qf_nx.mm(gf.t())
    res = scores.topk(5, dim=1)[1][:,0]
    top1correct = gl[res].eq(ql_nx).sum().item()

    print("Acc top1:{:.3f}".format(top1correct/ql.size(0))) # Acc top1:0.985
    pass

if __name__ == '__main__':
    model_path = "/workspace/py/deep_sort_pytorch/features_1.pth"
    model_path_nx_dir = "/workspace/py/deep_sort_pytorch/deep_sort/deep/checkpoint/nx_reid_result_int8"
    # pth models
    features = torch.load(model_path)
    evaluate(features)
    
    # # nx model results
    # # features_nx = torch.load(model_path_nx)
    # features_nx = load_json_data_and_to_torch(model_path_nx_dir)
    # evaluate(features_nx)

    # # compare two results
    # evaluate_two(features, features_nx)

