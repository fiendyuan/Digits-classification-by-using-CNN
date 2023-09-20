


# 测试一维卷积神经网络的相关代码

from keras.utils import plot_model
from keras import regularizers
from keras import Sequential
from keras.layers import *
from get_mfcc_1dcnn import *
from keras import optimizers
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt


#数据准备阶段

x_train =[]
y_train = []

# train1.txt 是自己录音的500条数据 quanbu.txt是老师提供的3000条录音加自己的500条录音进行训练
text = open('allmyvoice.txt','r',encoding='utf-8').read().split('\n')
for i in text:
    wenjian,leibie = i.split('	')

    x_train.append(get(wenjian))
    y_train.append((int(leibie)))



x_train = np.array(x_train)

y_train = np.array(y_train)
y_train = np_utils.to_categorical(y_train,10)
print(y_train)

model = Sequential()


# 1.h5  使用3000条原本加500条自己录音  使用的是get_mfcc中的 get()函数获取特征
# model.add(Conv1D(filters=16, kernel_size=5, strides=3,padding = 'same', input_shape = (1000, 1), activation = 'relu'))
# model.add(MaxPool1D(pool_size=2))
#
# model.add(Conv1D(filters=36,kernel_size=10,padding='same',activation='relu'))
# model.add(MaxPool1D(pool_size=2))
#
# model.add(Conv1D(filters=36,kernel_size=10,padding='same',activation='relu'))
# model.add(MaxPool1D(pool_size=2))
#
# model.add(Conv1D(filters=36,kernel_size=10,padding='same',activation='relu'))
# model.add(MaxPool1D(pool_size=2))

# 3.h5 使用500条自己录音  使用的是get_mfcc中的 get()函数获取特征
# model.add(Conv1D(filters=16, kernel_size=5, strides=3,padding = 'same', input_shape = (1000, 1), activation = 'relu'))
# model.add(MaxPool1D(pool_size=2))
#
# model.add(Conv1D(filters=36,kernel_size=10,padding='same',activation='relu'))
# model.add(MaxPool1D(pool_size=2))
# #
# model.add(Conv1D(filters=36,kernel_size=10,padding='same',activation='relu'))
# model.add(MaxPool1D(pool_size=2))
#
# model.add(Conv1D(filters=36,kernel_size=10,padding='same',activation='relu'))
# model.add(MaxPool1D(pool_size=2))

# 4.h5  使用的是500条自己的录音 使用的是get_mfcc中的 GetFrequencyFeature3（）函数获取特征
# model.add(Conv1D(filters=16, kernel_size=5, strides=3,padding = 'same', input_shape = (22500, 1), activation = 'relu'))
# model.add(MaxPool1D(pool_size=2))
# model.add(Conv1D(filters=36,kernel_size=5,padding='same',activation='relu'))
# model.add(MaxPool1D(pool_size=2))

# 5.h5 使用3000条原本加500条自己录音 ，使用的是get_mfcc中的 GetFrequencyFeature3（）函数获取特征
model.add(Conv1D(filters=16, kernel_size=5, strides=3,padding = 'same', input_shape = (1000, 1), activation = 'relu'))
# model.add(MaxPool1D(pool_size=2))
model.add(Conv1D(filters=32,kernel_size=5, strides=3, padding='same',activation='relu'))

model.add(MaxPool1D(pool_size=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()
opt = optimizers.adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
history=model.fit(x_train, y_train, batch_size=8, epochs=60)
model.save('allmyvoice.h5')

plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.savefig("Lossallmyvoice.png")
plt.show()

plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
plt.title('Model acc')
plt.ylabel('Acc')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.savefig("Accallmyvoice.png")
plt.show()