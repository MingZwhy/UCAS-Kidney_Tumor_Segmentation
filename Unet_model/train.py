import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import glob

import Unet_mode

# 读取图像
def read_png(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels = 3)
    img = tf.image.resize(img,(256,256))
    return img

# 读取标签
def read_png_label(path):
    mask = tf.io.read_file(path)
    mask = tf.image.decode_png(mask, channels = 1)
    mask = tf.image.resize(mask,(256,256))
    return mask

#对图像和标签做标准化
def normalize(img, mask):
    img = tf.cast(img, tf.float32)
    #归一化到(0,1)之间
    img = img / 127.5 - 1
    #标注就是简单的分类，int32足够
    mask = tf.cast(mask, tf.int8)
    return img, mask


# 批处理训练图+标注的函数
def load_image(img_path, mask_path):
    img = read_png(img_path)
    mask = read_png_label(mask_path)
    img, mask = normalize(img, mask)
    return img, mask

# 制作dataset
def make_dataset(image_dir_path, segemen_dir_path, BATCH_SIZE = 8,
                 SHUFF_SIZE = 200, train_ratio = 0.8):
    """
    由image和 segemen 制作dataset

    :param image_dir_path: 由处理nii图像所得的图像所在根目录。
    :param segemen_dir_path: 由处理nii图像所得的语义分割所在根目录。
    :param BATCH_SIZE: 训练的batch_size, 默认为8
    :param SHUFF_SIZE: 训练的buffer_size, 默认为200
    :param train_ratio: 训练集占比， 默认为0.8
    :return: train_ds, test_ds, step_per_epoch, val_step

    train_ds : 训练集
    test_ds : 测试集
    step_per_epoch : 训练步长
    val_step : 测试步长
    """

    if(image_dir_path[-1] != '/'):
        image_dir_path = image_dir_path + "/"
    if(segemen_dir_path[-1] != '/'):
        segemen_dir_path = segemen_dir_path + "/"

    image = glob.glob(image_dir_path + "*/*.png")
    label = glob.glob(segemen_dir_path + "*/*.png")

    length = 0
    if(len(image) != len(label)):
        print("wrong dir_path")
    else:
        length = len(image)

    # 打乱序列
    index = np.random.permutation(length)
    image = np.array(image)[index]
    label = np.array(label)[index]

    # 制作整体dataset
    all_ds = tf.data.Dataset.from_tensor_slices((image, label))
    all_ds = all_ds.map(load_image)

    # 划分train_dataset 和 test_dataset
    train_count = int(length * train_ratio)
    test_count = length - train_count

    step_per_epoch = train_count // BATCH_SIZE  # 训练步长
    val_step = test_count // BATCH_SIZE         # 测试步长

    train_ds = all_ds.skip(test_count)
    test_ds = all_ds.take(test_count)

    #对训练集 shuffle + batch
    #对测试集 batch
    train_ds = train_ds.shuffle(SHUFF_SIZE).batch(BATCH_SIZE)
    test_ds = test_ds.batch(BATCH_SIZE)

    return train_ds, test_ds, step_per_epoch, val_step

#定义IOU评估
class MeanIOU(tf.keras.metrics.MeanIoU):   #父类的方法需要y_true 和 y_pred有相同形状
    def __call__(self, y_true, y_pred, sample_weight=None):    #调用时默认会寻找call方法
        #y_pred原本是长度为3张量
        y_pred = tf.argmax(y_pred,axis=-1)
        return super().__call__(y_true, y_pred,sample_weight=sample_weight)

#单步训练
def train_step(images, labels, model,
               opt, loss_fn,
               train_loss, train_acc, train_iou):
    with tf.GradientTape() as tape:
        # 调用模型得到预测结果
        predictions = model(images)
        # 对比预测结果与标注得到损失
        loss = loss_fn(labels, predictions)

    # 用经典反向传播得到梯度
    gradients = tape.gradient(loss, model.trainable_variables)
    # 应用梯度对参数做优化
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_acc(labels, predictions)
    train_iou(labels, predictions)


def test_step(images, labels, model,
              loss_fn,
              test_loss, test_acc, test_iou):
    predictions = model(images)
    loss = loss_fn(labels, predictions)

    test_loss(loss)
    test_acc(labels, predictions)
    test_iou(labels, predictions)

def train_mode(train_ds, test_ds, step_per_epoch, val_step,
               save_model_dir_path, if_save = True, learn_rate = 0.0001, epochs = 50):
    """
    训练模型并保存训练好的模型
    :param train_ds: 训练集
    :param test_ds: 测试集
    :param step_per_epoch: 训练步长
    :param val_step: 测试步长
    :param save_model_dir_path: 模型参数保存根目录
    :param if_save: 是否保存训练好的模型的参数
    :param learn_rate: 学习速率，默认为0.0001
    :param epochs: 训练的轮数，默认为50
    return: model
    """

    model = Unet_mode.Unet_model()  # 实例化定义好的U-NET模型

    #定义优化器opt
    opt = tf.keras.optimizers.Adam(learn_rate)
    #定义损失函数
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
    train_iou = MeanIOU(3, name='train_iou')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')
    test_iou = MeanIOU(3, name='test_iou')

    for epoch in range(epochs):
        train_index = 1
        test_index = 1
        # 在下一个epoch开始时，重置评估指标
        train_loss.reset_states()
        train_acc.reset_states()
        train_iou.reset_states()
        test_loss.reset_states()
        test_acc.reset_states()
        test_iou.reset_states()

        print("training ", epoch + 1, " epoch: waiting......")
        for images, labels in train_ds:
            if (train_index % 50 == 0):
                print("batch ", train_index, "/", step_per_epoch)
            train_index = train_index + 1
            train_step(images, labels, model,
                       opt, loss_fn,
                       train_loss, train_acc, train_iou)

        print("testing......")
        for images, labels in test_ds:
            if (test_index % 50 == 0):
                print("batch ", test_index, "/", val_step)
            test_index = test_index + 1
            test_step(images, labels, model,
                      loss_fn,
                      test_loss, test_acc, test_iou)

        print("the epoch", epoch + 1, " result: ")
        template1 = "train --> Loss: {:.2f}, Accuracy: {:.2f}, IOU: {:.2f}"
        print(template1.format(train_loss.result(), train_acc.result() * 100, train_iou.result()))
        template2 = "test  --> Loss: {:.2f}, Accuracy: {:.2f}, IOU: {:.2f}"
        print(template2.format(test_loss.result(), test_acc.result() * 100, test_iou.result()))

    print("finish all ", epochs, " epochs!")\

    if(if_save):
        print("now will save the model trained")
        if(save_model_dir_path[-1] != '/'):
            save_model_dir_path = save_model_dir_path + "/"
        model.save_weights(save_model_dir_path + "Unet_model_weights.h5")
        print("save successfully in " + save_model_dir_path)

    return model



