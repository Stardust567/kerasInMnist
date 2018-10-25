from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.models import model_from_yaml
from keras.utils import np_utils
from keras.datasets import mnist
import numpy as np
import random

def dataSet(x_train, y_train):
    x_train = x_train.reshape(len(x_train), -1)
    x_train = x_train / 256
    y_train = np_utils.to_categorical(y_train, 10)
    '''转换为one_hot类型
        多分类cnn网络的输出通常是softmax层，为一个概率分布，
        要求输入的标签也以概率分布的形式出现，进而计算交叉熵'''
    return x_train, y_train

def dataLoad():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = dataSet(x_train, y_train)
    x_test, y_test = dataSet(x_test, y_test)
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = dataLoad()

    model = Sequential()
    model.add(Dense(input_dim=28 * 28,
                    units=500))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(units=500))
    model.add(Activation('relu'))
    model.add(Dropout(0.6))
    model.add(Dense(units=10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=100, nb_epoch=10)

    filename = "module_contain_weights.h5"
    model.save_weights(filename)
    # model.load_weights(filename, by_name=True)
    fit = model.evaluate(x_train, y_train)  # loss;accuracy
    score = model.evaluate(x_test,y_test) # loss;accuracy
    print(fit, score)
    '''results = model.predict(x_test)
'''


