from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.models import model_from_yaml
from keras.datasets import mnist
import numpy as np
import random

def dataSet(x_train, labels):
    x_train = x_train.reshape(len(x_train), -1)
    x_train = x_train / 256
    y_train = []
    for y in labels:
        y_label = np.zeros([10])
        y_label[y] = 1
        y_train.append(y_label)
    y_train = np.array(y_train)
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
    model.add(Dropout(0.5))
    model.add(Dense(units=500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
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


