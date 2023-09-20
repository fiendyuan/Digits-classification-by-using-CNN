import os
import csv
import time
import joblib
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import MinMaxScaler # 归一化
from sklearn.model_selection import train_test_split
from keras import regularizers  # 正则化
from keras.models import Sequential
from keras.layers import Dense, Dropout, advanced_activations, Conv1D, Flatten
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

def load_data(path):
    data = list()
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    data = np.array(data)
    f.close()
    return data

train_path = 'data/train_data.csv'
test_path = 'data/val_data.csv'

train_data = load_data(train_path)
test_data = load_data(test_path)

train_x = train_data[1:, 1:306]
new_train_y = train_data[1:, 306]
test_x = test_data[1:, 1:306]
new_test_y = test_data[1:, 306]

train_y = list()
test_y = list()

for one_train_y in new_train_y:
    new_one_train_y = list()

    if int(one_train_y[0]) == 0:
        new_one_train_y.append(1)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
    elif int(one_train_y[0]) == 1:
        new_one_train_y.append(0)
        new_one_train_y.append(1)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
    elif int(one_train_y[0]) == 2:
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(1)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
    elif int(one_train_y[0]) == 3:
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(1)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
    elif int(one_train_y[0]) == 4:
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(1)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
    elif int(one_train_y[0]) == 5:
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(1)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
    elif int(one_train_y[0]) == 6:
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(1)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
    elif int(one_train_y[0]) == 7:
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(1)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
    elif int(one_train_y[0]) == 8:
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(1)
        new_one_train_y.append(0)
    elif int(one_train_y[0]) == 9:
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(0)
        new_one_train_y.append(1)

    train_y.append(new_one_train_y)

for one_test_y in new_test_y:
    new_one_test_y = list()

    if int(one_test_y[0]) == 0:
        new_one_test_y.append(1)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
    elif int(one_test_y[0]) == 1:
        new_one_test_y.append(0)
        new_one_test_y.append(1)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
    elif int(one_test_y[0]) == 2:
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(1)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
    elif int(one_test_y[0]) == 3:
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(1)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
    elif int(one_test_y[0]) == 4:
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(1)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
    elif int(one_test_y[0]) == 5:
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(1)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
    elif int(one_test_y[0]) == 6:
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(1)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
    elif int(one_test_y[0]) == 7:
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(1)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
    elif int(one_test_y[0]) == 8:
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(1)
        new_one_test_y.append(0)
    elif int(one_test_y[0]) == 9:
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(0)
        new_one_test_y.append(1)

    test_y.append(new_one_test_y)

train_y = np.array(train_y)
test_y = np.array(test_y)
print(train_y.shape)
print(test_y.shape)

# wzj_X = list()
# wzj_Y = list()
# random_index = random.sample(range(0, len(X)), 100000)
# for i in random_index:
#     wzj_X.append(X[i])
#     wzj_Y.append(Y[i])
# wzj_X = np.array(wzj_X)
# wzj_Y = np.array(wzj_Y)
# X_train, X_test, Y_train, Y_test = train_test_split(wzj_X, wzj_Y, test_size=0.1)

x_train = pd.DataFrame(train_x)
x_test = pd.DataFrame(test_x)

min_max_scaler = MinMaxScaler()

scaler = min_max_scaler.fit(x_train)
joblib.dump(scaler, "model/scaler.m")
x_train = min_max_scaler.transform(x_train)
x_test = min_max_scaler.transform(x_test)

x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
print(x_train.shape)

model = Sequential()
# 第一层
# model.add(Dense(units = 10,activation='relu',input_shape=(x_train.shape[1],)))
model.add(Conv1D(filters=16,
                 kernel_size=10,
                 padding = 'same',
                 input_shape = (x_train.shape[1], 1),
                 activation = 'relu'))
# 第二层
model.add(Flatten())
# 第三层
model.add(Dense(units=10,activation='softmax'))

model.summary()

time_begin = time.time()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=["acc"])
model_checkpoint = ModelCheckpoint("model/best_model.h5", monitor='val_acc', verbose=1, save_best_only=True)

## 记录
history = model.fit(x_train, train_y,
          steps_per_epoch=len(x_train) // 32,
          epochs=10,
          #batch_size=1,
          verbose=2,
          validation_data = (x_test, test_y),
          validation_steps=len(x_test) // 32,
          callbacks=[model_checkpoint]
        )

predicts = model.predict(x_test)
print(predicts)
print(test_y)
print(np.argmax(predicts, axis=1))

##画图
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.savefig("val_loss.png")
plt.show()