import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



nx_output_g_concat_path = "/workspace/py/deep_sort_pytorch/results_analysis/nx_reid_result/output_g_concat.json"
with open(nx_output_g_concat_path,'r') as file_object:
    nx_output_g_concat = json.load(file_object)

nx_output_g_concat_list = np.array( 
    nx_output_g_concat['output_g_concat'] 
)           # 19732 总数，其中，19115 个为 0 


server_output_g_concat_path = "/workspace/py/deep_sort_pytorch/results_analysis/reid_reslt_server/output_g_concat_server.json"
with open(server_output_g_concat_path,'r') as file_object:
    server_output_g_concat = json.load(file_object)

server_output_g_concat_list = np.array( 
    server_output_g_concat['output_g_concat']
)
# 统计 0 的个数

# def  cout_zero_nums(a_numpy):
#      new_np = np.ceil(np.dot(np.abs(a_numpy, np.ones(a_numpy.shape[1])))).astype(int)
#      count_zero = sum(new_np==0)
#      return count_zero



# nx_output_g_concat_list_cout = cout_zero_nums(nx_output_g_concat_list)


# print("=====")


def cos_example(t1, t2):
    # t1  = np.array([-0.4,0.8,0.5,-0.2,0.3])
    # t2  = np.array([-0.5,0.4,-0.2,0.7,-0.1])

    def cos_sim(a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        cos = np.dot(a,b)/(a_norm * b_norm)
        return cos
    print(cos_sim(t1,t2))

# import numpy as np


cos_example(np.array(nx_output_g_concat_list), 
np.array(server_output_g_concat_list))



# a1=np.arange(15).reshape(3,5)
# a2=np.arange(20).reshape(4,5)

# print("cosine_similarity(a1,a2)")
# print(cosine_similarity(a1,a2))   #第一行的值是a1中的第一个行向量与a2中所有的行向量之间的余弦相似度

# print(cosine_similarity(a1))   #a1中的行向量之间的两两余弦相似度



# print("output_g_concat_list(a1,a2)")
# print(cosine_similarity(np.array(nx_output_g_concat_list), 
# np.array(server_output_g_concat_list))
# ) 



# ###################################################   numpy 中是否为0 

# -*- coding: utf-8 -*-
# @Time  : 2018/5/17 15:05
# @Author : Sizer
# @Site  : 
# @File  : test.py
# @Software: PyCharm
# import time
# import numpy as np
 
# data = np.array([
# [5.0, 3.0, 4.0, 4.0, 0.0],
# [3.0, 1.0, 2.0, 3.0, 3.0],
# [4.0, 3.0, 4.0, 3.0, 5.0],
# [0.0, 0.0, 0.0, 0.0, 0.0],
# [1.0, 5.0, 5.0, 2.0, 1.0],
# [0,   0,   0,   0,   0, ],
# ])
# # data = np.random.random((1000, 1000))
# print(data.shape)
# start_time = time.time()
# # avg = [float(np.mean(data[i, :])) for i in range(data.shape[0])]
# # print(avg)
 
 
# start_time = time.time()
# avg = []
# for i in range(data.shape[0]):
#   sum = 0
#   cnt = 0
#   for rx in data[i, :]:
#    if rx > 0:
#      sum += rx
#      cnt += 1
#   if cnt > 0:
#    avg.append(sum/cnt)
#   else:
#    avg.append(0)
# end_time = time.time()
# print("op 1:", end_time - start_time)
 
# start_time = time.time()
# avg = []
# isexist = (data > 0) * 1
# for i in range(data.shape[0]):
#   sum = np.dot(data[i, :], isexist[i, :])
#   cnt = np.sum(isexist[i, :])
#   if cnt > 0:
#    avg.append(sum / cnt)
#   else:
#    avg.append(0)
# end_time = time.time()
# print("op 2:", end_time - start_time)
# #
# # print(avg)
# factor = np.mat(np.ones(data.shape[1])).T
# # print("facotr :")
# # print(factor)
# exist = np.mat((data > 0) * 1.0)
# # print("exist :")
# # print(exist)
# # print("res  :")
# res = np.array(exist * factor)
# end_time = time.time()
# print("op 3:", end_time-start_time)
 
# start_time = time.time()
# exist = (data > 0) * 1.0
# factor = np.ones(data.shape[1])
# res = np.dot(exist, factor)
# end_time = time.time()
# print("op 4:", end_time - start_time)

# ####################################################   END: numpy 中是否为0 


