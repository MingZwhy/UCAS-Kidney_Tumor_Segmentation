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
import evaluate

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
        "-a", "--if_process_data", required=True, default="True",
        help="if you run it first, you should choose True so that it will process the data automatically"
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
    parser.add_argument(
        "-m", "--model_kind", required=False, default="FCN_model",
        help="choose the train model"
    )
    parser.add_argument(
        "-d", "--if_predict", required=False, default="False",
        help="whether to predirc data without label"
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
    if (args.if_process_data == "True" and os.path.exists(load_nii_dir_path) == False):
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
    if(args.if_process_data == "True"):
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
    train_ds, test_ds, evaluate_ds, step_per_epoch, val_step = train.make_dataset(
        image_dir_path = save_image_dir_path,
        segemen_dir_path = save_segemen_dir_path,
        BATCH_SIZE = 8,
        SHUFF_SIZE = 200,
        train_ratio = 0.7,
        test_ratio = 0.2,
        eva_ratio = 0.1,
        model_kind = args.model_kind)

    #训练model
    model = train.train_mode(train_ds, test_ds, step_per_epoch, val_step, args.model_kind,
                             save_model_dir_path, if_save, learn_rate, epochs)

    #在检验集上对结果进行评估
    evaluate.evaluate_model(model, evaluate_ds, args.model_kind)

    #对无标签图像进行预测
    if(args.if_predict == "True"):
        predict.predict_and_save_result(model, save_evaluate_image_dir_path, predict_result_dir_path)