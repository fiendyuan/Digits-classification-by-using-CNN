
from keras.models import load_model

from get_mfcc_1dcnn import *

model = load_model('4.h5')

m=0
n=0
for i in open('train1.txt','r',encoding='utf-8').read().split('\n'):
    file = i.split('	')[0]
    yuyin=i.split('	')[-1]
    print(file)
    x_test = get(file).reshape((1, 1000, 1))
    result = model.predict(x_test)[0].tolist()
    j = result.index(max(result))
    print(type(j))
    m=m+1
    if str(j)==yuyin:
        n=n+1
    print(n)
print('识别正确的语音总个数：' + str(n))
print('语音总个数：'+str(m))
print('训练集识别的正确率：'+str(float(n/m)))