from keras.models import Sequential
import test

if __name__ == '__main__':
    model = Sequential()

    filename = "module_contain_weights.h5"
    model.load_weights(filename, by_name=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    x_train, y_train, x_test, y_test = test.dataLoad()
    score = model.evaluate(x_test, y_test)
    print(score)