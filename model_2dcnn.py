from keras import Sequential
from keras.layers import *
from get_mfcc_2dcnn import *
from keras import optimizers
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint


def get_list():
    x_train = []
    y_train = []

    # train1.txt 是自己录音的500条数据 quanbu.txt是老师提供的3000条录音加自己的500条录音进行训练
    text = open('train_new.txt', 'r', encoding='utf-8').read().split('\n')
    for i in text:
        wenjian, leibie = i.split('	')

        x_train.append(get(wenjian))
        y_train.append((int(leibie)))

    x_train = np.array(x_train)

    y_train = np.array(y_train)
    y_train = np_utils.to_categorical(y_train, 10)

    x_test = []
    y_test = []

    # train1.txt 是自己录音的500条数据 quanbu.txt是老师提供的3000条录音加自己的500条录音进行训练
    text2 = open('test_new.txt', 'r', encoding='utf-8').read().split('\n')
    for i in text2:
        wenjian, leibie = i.split('	')

        x_test.append(get(wenjian))
        y_test.append((int(leibie)))

    x_test = np.array(x_test)

    y_test = np.array(y_test)
    y_test = np_utils.to_categorical(y_test, 10)

    return x_train,y_train,x_test,y_test

def train_model(x_train, y_train, x_ver, y_ver):
    '''

    :param x_train: x训练集
    :param y_train: y训练集
    :param x_ver: x验证集
    :param y_ver: y验证集
    :return:
    '''
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=(20, 50, 1), activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Dense(100, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.summary()
    opt = optimizers.adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # history = model.fit(x_train, y_train, batch_size=16, epochs=10)
    model_checkpoint = ModelCheckpoint("model/2dcnn/new1.h5", monitor='val_acc', verbose=1, save_best_only=True)

    history = model.fit(x_train, y_train,

                        batch_size=16,  # 每批次放入模型中训练的数据
                        epochs=200,  # 训练的轮数
                        validation_data=(x_ver, y_ver),
                        shuffle=True,
                        callbacks=[model_checkpoint])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Ver'], loc='upper right')
    plt.savefig("Loss2dcnn.png")
    plt.show()

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model acc')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Ver'], loc='upper right')
    plt.savefig("Acc2dcnn.png")
    plt.show()

if __name__ =='__main__':
    x_train, y_train, x_test, y_test=get_list()
    train_model(x_train, y_train, x_test, y_test)