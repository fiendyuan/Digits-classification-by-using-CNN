

# import os
# list= []
# file1 = open('quanbu.txt',mode='a+',encoding='utf-8')
# for i in os.listdir('voices/voices/'):
#     shuzi=i.split('_')[0]
#     list.append('voices/voices/'+i+'	'+shuzi)
#
#
# jie49.wav
# jie30.wav
# jie12.wav
# jie1.wav
# shuzi=i.split('.')[0].split('e')[-1]
# i=0_george_15.wav
# i.split('_') = ['0','george','15.wav']

import numpy as np
# import random
# random.shuffle(list)

# for i in list:
#     filename=i
#     shuzi=i.split('_')[0]
#     file1.write(i + '\n')


# import wave
# import os
# list=[]
# for i in os.listdir('recordings/'):
#
#     wav = wave.open('recordings/'+i)
#     num_frame = wav.getnframes()
#     list.append(num_frame)
#     print(num_frame)
#
# print(max(list))
import os

list= []
file1 = open('allmyvoice.txt',mode='w',encoding='utf-8')
for i in os.listdir('voice/voices/'):
    shuzi=i.split('_')[0]
    # list.append('voice/voices/'+i+'	'+shuzi)
    file1.write('voice/voices/'+ i +'\t'+ shuzi +'\n')

