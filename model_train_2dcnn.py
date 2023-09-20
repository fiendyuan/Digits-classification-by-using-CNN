


# Testing related code for 2D convolutional neural networks

from keras.utils import plot_model
from keras import regularizers
from keras import Sequential
from keras.layers import *
from get_mfcc_2dcnn import *
from keras import optimizers
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt

x_train =[]
y_train = []


# train1.txt contains 500 recordings made by oneself, while quanbu.txt contains 3000 recordings provided by the teacher plus the 500 recordings made by oneself for training.
text = open('3500.txt','r',encoding='utf-8').read().split('\n')
for i in text:
    wenjian,leibie = i.split('	')

    x_train.append(get(wenjian))
    y_train.append((int(leibie)))



x_train = np.array(x_train)

y_train = np.array(y_train)
y_train = np_utils.to_categorical(y_train,10)
print(y_train)

model = Sequential()

#  Training a model using 500 of my own recorded audio clips and extracting features using the 'GetFrequencyFeature3()' function from 'get_mfcc_2dcnn'.
model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',input_shape=(50, 20, 1),activation='relu'))
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Dense(100,activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()
opt = optimizers.adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
history=model.fit(x_train, y_train, batch_size=16, epochs=10)
model.save('model/2dcnn/7.h5')


plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.savefig("val_loss.png")
plt.show()
# plt.figure()
# plt.plot(history.history["loss"], label="train_loss")
# plt.plot(history.history["val_loss"], label="val_loss")
# plt.plot(history.history["acc"], label="train_acc")
# plt.plot(history.history["val_acc"], label="val_acc")
# plt.title("Training Loss and Accuracy")