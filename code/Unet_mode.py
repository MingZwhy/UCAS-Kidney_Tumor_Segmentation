import tensorflow as tf
from tensorflow import keras

#搭建Unet模型

"""
调用方法：
    import Unet_mode
    model = Unet_model()
"""

# 封装下采样
class DownSample(tf.keras.layers.Layer):
    def __init__(self, units):
        super(DownSample, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(units, kernel_size=3,
                                            padding='same')
        self.conv2 = tf.keras.layers.Conv2D(units, kernel_size=3,
                                            padding='same')
        self.pool = tf.keras.layers.MaxPooling2D()

    def call(self, x, Is_Pool=True):
        if (Is_Pool):
            x = self.pool(x)  # 下采样
        x = self.conv1(x)
        x = tf.nn.relu(x)  # 采样后需要激活
        x = self.conv2(x)
        x = tf.nn.relu(x)
        return x


# 封装上采样
class UpSample(tf.keras.layers.Layer):
    def __init__(self, units):
        super(UpSample, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(units, kernel_size=3,
                                            padding='same')
        self.conv2 = tf.keras.layers.Conv2D(units, kernel_size=3,
                                            padding='same')
        self.deconv = tf.keras.layers.Conv2DTranspose(units // 2, kernel_size=2,
                                                      strides=2, padding='same')
        # 反卷积上采样时单元数减半(units//2),图像大小加倍(strides=2)

    def call(self, x):
        x = self.conv1(x)
        x = tf.nn.relu(x)  # 采样后需要激活
        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = self.deconv(x)
        x = tf.nn.relu(x)
        return x

# Unet_model
class Unet_model(tf.keras.Model):
    def __init__(self):
        super(Unet_model, self).__init__()
        self.down1 = DownSample(64)
        self.down2 = DownSample(128)
        self.down3 = DownSample(256)
        self.down4 = DownSample(512)
        self.down5 = DownSample(1024)

        self.middle_up = tf.keras.layers.Conv2DTranspose(512, kernel_size=2,
                                                         strides=2, padding='same')

        self.up1 = UpSample(512)
        self.up2 = UpSample(256)
        self.up3 = UpSample(128)

        self.conv_last = DownSample(64)    # Is_pool=False

        self.last = tf.keras.layers.Conv2D(3,
                                           kernel_size=1,
                                           padding='same')
        # 语义分割本质上是分类问题，分3类则最终输出3个通道

    # 前向传播
    def call(self, x):
        x1 = self.down1(x, Is_Pool=False)  # 初始第一个是没有下采样的
        x2 = self.down2(x1)                # 这里调用的是其call方法
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x5 = self.middle_up(x5)

        x5 = tf.concat([x4, x5], axis=-1)  # 在通道维度上合并，同尺寸，512+512得到共1024通道
        x5 = self.up1(x5)

        x5 = tf.concat([x3, x5], axis=-1)
        x5 = self.up2(x5)

        x5 = tf.concat([x2, x5], axis=-1)
        x5 = self.up3(x5)

        x5 = tf.concat([x1, x5], axis=-1)
        x5 = self.conv_last(x5, Is_Pool=False)

        x5 = self.last(x5)

        return x5

#model = Unet_model()
#model.build(input_shape =(None,256,256,3))
#model.summary()
