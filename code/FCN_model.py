import tensorflow as tf
from tensorflow import keras

#基于VGG16搭建FCN模型

"""
调用方法：
    import FCN_model
    model = FCN_model()
"""

def FCN_model():

    conv_base = tf.keras.applications.VGG16(weights='imagenet',
                                           input_shape=(256,256,3),
                                           include_top=False)

    #out_shape: 7,7,512
    #对它做上采样——14,14,512
    #之后令其与block5_conv3的结果14,14,512相加
    #相加结果再做上采样——28,28,512，与block4_conv3的结果28,28,512相加
    #上采样——56,56,256与block3_conv3的结果相加
    #...

    layer_names = [
        'block5_conv3',
        'block4_conv3',
        'block3_conv3',
        'block5_pool'
    ]
    layers_output = [conv_base.get_layer(layer_name).output for layer_name in layer_names]

    multi_out_model = tf.keras.models.Model(inputs=conv_base.input,
                                           outputs=layers_output)
    #这里的输出是一个列表，包含了我们后面要用到的4个层
    multi_out_model.trainable = False

    inputs = tf.keras.Input(shape=(256,256,3))
    out_block5_conv3, out_block4_conv3, out_block3_conv3, out = multi_out_model(inputs)

    x1 = tf.keras.layers.Conv2DTranspose(512,3,strides=2,padding='same',activation='relu')(out)
    #加一层不改变形状的卷积提取特征
    x1 = tf.keras.layers.Conv2D(512,3,strides=1,padding='same',activation='relu')(x1)

    x2 = tf.add(x1,out_block5_conv3)
    x2 = tf.keras.layers.Conv2DTranspose(512,3,strides=2,padding='same',activation='relu')(x2)
    #加一层不改变形状的卷积提取特征
    x2 = tf.keras.layers.Conv2D(512,3,strides=1,padding='same',activation='relu')(x2)

    x3 = tf.add(x2,out_block4_conv3)
    x3 = tf.keras.layers.Conv2DTranspose(256,3,strides=2,padding='same',activation='relu')(x3)
    #加一层不改变形状的卷积提取特征
    x3 = tf.keras.layers.Conv2D(256,3,strides=1,padding='same',activation='relu')(x3)

    x4 = tf.add(x3,out_block3_conv3)
    x4 = tf.keras.layers.Conv2DTranspose(128,3,strides=2,padding='same',activation='relu')(x4)
    x4 = tf.keras.layers.Conv2D(128,3,strides=1,padding='same',activation='relu')(x4)

    prediction = tf.keras.layers.Conv2DTranspose(3,3,strides=2,padding='same',activation='softmax')(x4)   #本质上是一个3分类问题

    model = tf.keras.models.Model(inputs=inputs,
                                 outputs=prediction)

    return model