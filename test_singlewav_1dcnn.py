
# 1.h5是3000条原本加500条自己录音
from keras.models import load_model

from get_mfcc_1dcnn import *
import time
model = load_model('4.h5')


import os
for i in os.listdir('xin/'):
    file='xin/'+i
    x_test = get(file).reshape((1, 1000, 1))
    # x_test = get(file).reshape((1, 1000, 1))
    result = model.predict(x_test)[0].tolist()
    # print(result)
    j = result.index(max(result))

    print('识别的结果为：'+str(j))