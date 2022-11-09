import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os

#for Parse command line arguments
import argparse

import dataset
import Unet_mode
import train
import predict

if __name__ == '__main__':
    # Parse command line arguments
    desc = "Choose whether save weights or not, learn_rate and epochs," \
           "as for other params like paths, please change them in py directly"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "-p", "--nii_data_dir_path", required=True, default="D:/lumor_segementation/kits19-master/",
        help="must tell py where your nii_data file are (path), for example: D:/lumor_segementation/kits19-master/"
    )
    parser.add_argument(
        "-i", "--if_save_weights", required=False, default="True",
        help="if true, py will save model_weights in save_model_dir_path"
    )
    parser.add_argument(
        "-lr", "--learn_rate", required=False, default=0.0001,
        help="choose the learn_rate"
    )
    parser.add_argument(
        "-e", "--train_epochs", required=False, default=20,
        help="choose the num of epochs for train model"
    )
    args = parser.parse_args()

    #check whether load_nii_dir_path existed or not
    dir_path = args.nii_data_dir_path

    if(dir_path[-1] != '/'):
        dir_path = dir_path + "/"

    load_nii_dir_path = dir_path + "data/"
    save_image_dir_path = dir_path + "p_image/"
    save_segemen_dir_path = dir_path + "p_segemen/"
    save_model_dir_path = dir_path + "model_weights/"

    save_evaluate_image_dir_path = dir_path + "evaluate_image/"
    predict_result_dir_path = dir_path + "predict_result/"

    # 检查路径 并 创建子目录
    if (os.path.exists(load_nii_dir_path) == False):
        print("wrong nii_data_dir_path or name of nii_file is not 'data'")
        print("please check your path and name of nii_file then run again")
    else:
        if (os.path.exists(save_image_dir_path) == False):
            os.makedirs(save_image_dir_path)
        if (os.path.exists(save_segemen_dir_path) == False):
            os.makedirs(save_segemen_dir_path)
        if (os.path.exists(save_model_dir_path) == False):
            os.makedirs(save_model_dir_path)
        if (os.path.exists(save_evaluate_image_dir_path) == False):
            os.makedirs(save_evaluate_image_dir_path)
        if (os.path.exists(predict_result_dir_path) == False):
            os.makedirs(predict_result_dir_path)

    #处理nii文件
    dataset.process_haslabel_pic(load_nii_dir_path, save_image_dir_path,
                                 save_segemen_dir_path, 210)
    dataset.process_nolabel_pic(load_nii_dir_path, save_evaluate_image_dir_path,
                                begin_index = 210, num_of_nii = 18)

    if_save = True
    if(args.if_save_weights != "True"):
        if_save = False

    learn_rate = float(args.learn_rate)
    print(learn_rate)
    epochs = int(args.train_epochs)

    #制作dataset
    train_ds, test_ds, step_per_epoch, val_step = train.make_dataset(save_image_dir_path, save_segemen_dir_path)

    #训练model
    model = train.train_mode(train_ds, test_ds, step_per_epoch, val_step,
                             save_model_dir_path, if_save, learn_rate, epochs)

    #对无标签图像进行预测
    predict.predict_and_save_result(model, save_evaluate_image_dir_path, predict_result_dir_path)