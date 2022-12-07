import os.path
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('AGG')   #needn't show --> may cause exception
import matplotlib.pyplot as plt
import numpy as np
import glob
from train import read_png

def load_test_image(path):
    img = read_png(path)
    img = tf.cast(img, tf.float32)
    img = img / 127.5 + 1
    return img

def predict_and_save_result(model, save_evaluate_image_dir_path, predict_result_dir_path):
    """
    对无标签的图像做预测并保存预测结果
    :param model: 训练好的模型
    :param save_evaluate_image_dir_path: 无标签的图像
    :param predict_result_dir_path: 保存预测结果的路径
    :return:
    """

    img_path = glob.glob(save_evaluate_image_dir_path + "*/*.png")
    num_of_img = len(img_path)

    for index in range(num_of_img):
        print("predict %d" %index)
        img_decode = load_test_image(img_path[index])
        img_decode_ex = tf.expand_dims(img_decode,0)
        pred_mask = model.predict(img_decode_ex)
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[...,tf.newaxis]

        #save result
        #print(img_path[index])
        case_path = img_path[index].split("\\")[-2]
        #print(case_path)
        dir_case_path = predict_result_dir_path + case_path
        #print(dir_case_path)
        if(os.path.exists(dir_case_path) == False):
            os.makedirs(dir_case_path)

        name = "/predict_" + img_path[index].split("\\")[-1]
        complete_path = dir_case_path + name
        #print(complete_path)
        plt.imshow(pred_mask[0])
        plt.savefig(complete_path)