import tensorflow as tf
from tensorflow import keras

#搭建LinkNet模型

"""
调用方法：
    import Linknet_mode
    model = LinkNet()
"""

# 1:卷积模块
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, units, k_size=3, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(units,
                                           kernel_size=k_size,
                                           strides=stride,
                                           padding='same')
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x):
        x = self.conv(x)
        x = tf.nn.relu(self.bn(x))
        return x


# 2:反卷积模块
class DeConvBlock(tf.keras.layers.Layer):
    def __init__(self, units, k_size=3, stride=2):
        super(DeConvBlock, self).__init__()
        self.deconv = tf.keras.layers.Conv2DTranspose(units,
                                                      kernel_size=k_size,
                                                      strides=stride,
                                                      padding='same')
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x, is_act=True):
        x = self.deconv(x)
        if (is_act):
            x = tf.nn.relu(self.bn(x))
        return x


# 编码器模块
class EncodeBlock(tf.keras.layers.Layer):
    def __init__(self, units):
        super(EncodeBlock, self).__init__()
        self.conv1 = ConvBlock(units, k_size=3, stride=2)
        self.conv2 = ConvBlock(units, k_size=3, stride=1)
        self.conv3 = ConvBlock(units, k_size=3, stride=1)
        self.conv4 = ConvBlock(units, k_size=3, stride=1)

        self.shortcut = ConvBlock(units, k_size=1, stride=2)

    def call(self, x, is_act=True):
        out1 = self.conv1(x)
        out1 = self.conv2(out1)
        residue1 = self.shortcut(x)
        add1 = out1 + residue1
        out2 = self.conv3(add1)
        out2 = self.conv4(out2)
        add2 = add1 + out2
        return add2


# 解码器模块
class DecodeBlock(tf.keras.layers.Layer):
    def __init__(self, units1, units2):
        super(DecodeBlock, self).__init__()
        self.conv1 = ConvBlock(units1, k_size=1)
        self.deconv = DeConvBlock(units1)
        self.conv2 = ConvBlock(units2, k_size=1)

    def call(self, x):
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.conv2(x)
        return x


class LinkNet(tf.keras.Model):
    def __init__(self):
        super(LinkNet, self).__init__()
        self.init_conv = ConvBlock(64, k_size=7, stride=2)
        self.input_pool = tf.keras.layers.MaxPooling2D(padding='same')

        self.encode1 = EncodeBlock(64)
        self.encode2 = EncodeBlock(128)
        self.encode3 = EncodeBlock(256)
        self.encode4 = EncodeBlock(512)

        self.decode4 = DecodeBlock(512 // 4, 256)
        self.decode3 = DecodeBlock(256 // 4, 128)
        self.decode2 = DecodeBlock(128 // 4, 64)
        self.decode1 = DecodeBlock(64 // 4, 64)

        self.deconv_last1 = DeConvBlock(32)
        self.conv_last = ConvBlock(32)
        #最终分3类
        self.deconv_last2 = DeConvBlock(3, k_size=2)

    def call(self, x):
        x = self.init_conv(x)
        x = self.input_pool(x)

        e1 = self.encode1(x)
        e2 = self.encode2(e1)
        e3 = self.encode3(e2)
        e4 = self.encode4(e3)

        d4 = self.decode4(e4) + e3
        d3 = self.decode3(d4) + e2
        d2 = self.decode2(d3) + e1
        d1 = self.decode1(d2)

        f = self.deconv_last1(d1)
        f = self.conv_last(f)
        f = self.deconv_last2(f, is_act=False)  # 输出不被激活

        return f

#model = LinkNet()
#model.build(input_shape =(None,256,256,3))
#model.summary()