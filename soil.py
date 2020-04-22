from __future__ import print_function

import os

import keras
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import Input, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense, Dropout, Flatten
from keras.layers import MaxPool2D
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split

# 对环境，GPU的一些设置
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def fft(img):  # 定义高通滤波器，滤波后反变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    fshift[crow - 18:crow + 18, ccol - 18:ccol + 18] = 0

    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)
    return iimg


def huber_loss(y_true, y_pred):  # 定义损失函数
    return tf.losses.huber_loss(y_true, y_pred)


def pic_load(path):  # 读取图片并Resize，滤波
    name = [os.path.join(path, f) for f in os.listdir(path)]
    data = []
    for url in name:
        img = Image.open(url)
        img = img.convert("L")
        img = img.resize((200, 125), Image.ANTIALIAS)
        img = np.array(img)
        img = fft(img)
        data.append(img)
    return data


def data_utils(seed):  # 划分训练集，测试集
    pic = pic_load(r"/soil_20/data_20")
    lab = np.loadtxt(r"/soil_20/val.txt", delimiter=",")
    lst = [i for i in range(128)]
    lst2 = [i for i in range(128)]
    a, b, c, d = train_test_split(lst, lst2, test_size=0.3, random_state=seed)
    input1 = []
    label1 = []
    for i in a:
        input1.append(pic[i])
        label1.append(lab[i])
    input1 = np.array(input1)
    label1 = np.array(label1)
    input2 = []
    label2 = []
    for i in b:
        input2.append(pic[i])
        label2.append(lab[i])
    input2 = np.array(input2)
    label2 = np.array(label2)
    return input1, input2, label1, label2


def model_build():
    input = Input(shape=(125, 200, 1), name='input')
    x1 = Conv2D(32, (3, 3), strides=2)(input)
    x1 = Conv2D(64, (3, 3), strides=2)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = BatchNormalization(momentum=0.8)(x1)
    x1 = MaxPool2D(pool_size=2, strides=1)(x1)
    x1 = Flatten()(x1)

    x = Dense(100, activation='relu')(x1)
    x = Dropout(0.1)(x)
    x = Dense(10, activation="relu")(x)
    output = Dense(1, name='output')(x)
    model = Model(inputs=[input], outputs=[output])
    model.summary()
    return model


def my_generator(a, b, batch_size):
    total_size = len(b)
    while 1:
        for i in range(total_size // batch_size):
            yield ({'input': a[i * batch_size:(i + 1) * batch_size].reshape((batch_size, 125, 200, 1))},
                   {'output': b[i * batch_size:(i + 1) * batch_size]})


batch_size = 8
model = model_build()
input1, input2, label1, label2 = data_utils(111111111)
model.compile(loss=huber_loss, optimizer="adam")
checkpoint = ModelCheckpoint(r"best_weights.h5", monitor='val_loss',
                             save_weights_only=True, verbose=1, save_best_only=True, period=1)
if os.path.exists(r"best_weights.h5"):
    model.load_weights(r"best_weights.h5")
    # 若成功加载前面保存的参数，输出下列信息
    print("checkpoint_loaded")
model.fit_generator(my_generator(input1, label1, batch_size), steps_per_epoch=len(input1) // batch_size,
                    validation_data=my_generator(input2, label2, batch_size),
                    validation_steps=len(input2) // batch_size,
                    epochs=100, callbacks=[checkpoint])
model.save("best_weights.h5")
