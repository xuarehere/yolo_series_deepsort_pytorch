'''
Author: xuarehere
Date: 
LastEditTime: 
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: 
'''

"""

python prepare_data.py --data_dir {data_path}/Market-1501-v15.09.15
"""
import os
from shutil import copyfile
import argparse

# You only need to change this line to your dataset download path
# download_path = "/Market-1501/Market-1501-v15.09.15"
def process_data(dir_path=None):
    """[summary]
    数据解析
    Args:
        dir_path ([type], optional): [description]. Defaults to None.
        dir_path 目录结构如下
        dir_path/
        |-- bounding_box_test
        |-- bounding_box_train
        |-- gt_bbox
        |-- gt_query
        `-- query
        
        解析后的文件结构如下：
        dir_path/pytorch/
        |-- gallery
        |-- multi-query
        |-- query
        |-- train
        |-- train_all
        `-- val
    Raises:
        Exception: [description]
    """
    if dir_path is None or not os.path.isdir(dir_path):
        raise Exception("{} is not found".format(dir_path))
    download_path =    dir_path         
    nums = 0
    nums_dict = {}
    if not os.path.isdir(download_path):
        print('please change the download_path')

    save_path = download_path + '/pytorch'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    #-----------------------------------------
    #query
    query_path = download_path + '/query'
    query_save_path = download_path + '/pytorch/query'
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)

    for root, dirs, files in os.walk(query_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = query_path + '/' + name
            dst_path = query_save_path + '/' + ID[0] 
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)
            nums += 1
    print("query, finish, nums: {}".format(nums))
    nums_dict["query"] = nums

    #-----------------------------------------
    #multi-query
    nums = 0
    query_path = download_path + '/gt_bbox'
    # for dukemtmc-reid, we do not need multi-query
    if os.path.isdir(query_path):
        query_save_path = download_path + '/pytorch/multi-query'
        if not os.path.isdir(query_save_path):
            os.mkdir(query_save_path)

        for root, dirs, files in os.walk(query_path, topdown=True):
            for name in files:
                if not name[-3:]=='jpg':
                    continue
                ID  = name.split('_')
                src_path = query_path + '/' + name
                dst_path = query_save_path + '/' + ID[0]
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                copyfile(src_path, dst_path + '/' + name)
                nums += 1
    print("gt_bbox, finish, nums: {}".format(nums))
    nums_dict["multi-query"] = nums
    #-----------------------------------------
    #gallery
    nums = 0
    gallery_path = download_path + '/bounding_box_test'
    gallery_save_path = download_path + '/pytorch/gallery'
    if not os.path.isdir(gallery_save_path):
        os.mkdir(gallery_save_path)

    for root, dirs, files in os.walk(gallery_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = gallery_path + '/' + name
            dst_path = gallery_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)
            nums += 1
    print("bounding_box_test, finish, nums: {}".format(nums))
    nums_dict["gallery"] = nums
    #---------------------------------------
    #train_all
    nums = 0
    train_path = download_path + '/bounding_box_train'
    train_save_path = download_path + '/pytorch/train_all'
    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)

    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = train_path + '/' + name
            dst_path = train_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)
            nums += 1
    print("bounding_box_train, finish, nums: {}".format(nums))
    nums_dict["train_all"] = nums


    #---------------------------------------
    #train_val
    nums = 0
    nums_val = 0
    train_path = download_path + '/bounding_box_train'
    train_save_path = download_path + '/pytorch/train'
    val_save_path = download_path + '/pytorch/val'
    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)
        os.mkdir(val_save_path)

    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = train_path + '/' + name
            dst_path = train_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
                dst_path = val_save_path + '/' + ID[0]  #first image is used as val image
                os.mkdir(dst_path)
                nums_val +=1
            else:
                nums += 1
            copyfile(src_path, dst_path + '/' + name)
            
    print("val, finish, nums: {}".format(nums_val))
    nums_dict["val"] = nums_val

    print("train, finish, nums: {}".format(nums))
    nums_dict["train"] = nums
    

def main():
    
    parser = argparse.ArgumentParser(description="ReID dataset Market-1501 pre-process")
    parser.add_argument(
        "--data_dir", default="./Market-1501-v15.09.15", help="dataset directory", type=str
    )
    
    args = parser.parse_args()
    print(args)
    process_data(dir_path=args.data_dir)


if __name__ == '__main__':
    main()

# query, finish, nums: 3368
# gt_bbox, finish, nums: 25259
# bounding_box_test, finish, nums: 19732
# bounding_box_train, finish, nums: 12936
# val, finish, nums: 751
# train, finish, nums: 12185
