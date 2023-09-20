# -*- coding: utf-8 -*-
# 程序入口

# 标准GUI库
import tkinter as tk
# Universally Unique Identifier，设置唯一的辨识信息
import uuid
# 操作系统的封装
import os
# 将不可见字符转换为可见字符的编码方式
import base64
# 颜文字库

# 访问平台相关属性
import platform as plat
# 引用自己的文件

# TensorFlow官方的高层API，Keras是一个高层神经网络API，并对TensorFlow等有较好的优化
from keras import backend as K
# 进行录音，播放，生成wav文件
import pyaudio
# 音频文件操作
import wave
# 多线程操作
import threading
# 对python基本类型值与用python字符串格式表示的C struct类型间的转化
import struct
from keras.models import load_model

from get_mfcc_1dcnn import *


def press_button_record(filename):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 8000
    RECORD_SECONDS = 2
    # RECORDID = str(self.stu_num)+'_'+str(num).zfill(2)+'_'+str(iter).zfill(2)
    WAVE_OUTPUT_FILENAME =  filename
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording:")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Recorded.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))



def hit_me():
    # 用global使用定义在函数外的变量的值
    global on_hit
    # global wave_name
    # 开启录音
    if on_hit == False:
        b.config(image=img_stop)
        # 构成完整文件存储路径
        # wave_name = f'{tmp_file_path}/tmp_file_{uuid.uuid4()}.wav'
        wave_name = 'tmp_wav/'+'my.wav'
        press_button_record(wave_name)

        x_test = get(wave_name).reshape((1, 1000, 1))
        model = load_model('1.h5')
        result = model.predict(x_test)[0].tolist()
        j = result.index(max(result))


        var.set(j)
        on_hit = False

        b.config(image=img_start)
# def hit_me():
#     CHUNK = 1024
#     FORMAT = pyaudio.paInt16
#     CHANNELS = 1
#     RATE = 8000
#     RECORD_SECONDS = 2
#     # RECORDID = str(self.stu_num)+'_'+str(num).zfill(2)+'_'+str(iter).zfill(2)
#     WAVE_OUTPUT_FILENAME = 'voice/' + 'my.wav'
#     p = pyaudio.PyAudio()
#     stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
#     print("Recording:")
#     frames = []
#     for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#         data = stream.read(CHUNK)
#         frames.append(data)
#     print("Recorded.")
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#     with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
#         wf.setnchannels(CHANNELS)
#         wf.setsampwidth(p.get_sample_size(FORMAT))
#         wf.setframerate(RATE)
#         wf.writeframes(b''.join(frames))
#     print('录音结束')

'''
主程序 界面绘制
'''
# GUI工具包的接口
window = tk.Tk()
window.title('孤立词识别')
window.geometry('560x450')
# 跟踪变量的值的变化
var = tk.StringVar()
# 新建Label标签控件，显示一行或多行文本且不允许用户修改，relief参数指定边框样式，anchor参数控制文本在 Label 中显示的位置
label_ = tk.Label(window, textvariable=var, borderwidth=1, relief="sunken", font=('宋体', 12),
             width=50, height=20, anchor='nw', bg="#FFCCCC")
# Pack为一布局管理器，可将它视为一个弹性的容器
label_.pack()

# 定义录音按钮图片
img_start = tk.PhotoImage(file="record.gif")
# 定义录音停止按钮图片
img_stop = tk.PhotoImage(file="stop.gif")


is_playing = False
my_thread = None
on_hit = False
tmp_file_path = 'tmp_wav'

# 文字输出内容
string_list = []

# 判断tmp_file_path路径是否存在，不存在创建路径
if not os.path.exists(tmp_file_path):
    os.mkdir(tmp_file_path)

# 新建Button控件
b = tk.Button(window, text='hit me', borderwidth=0, command=hit_me)

# 设置按钮图片
b.config(image=img_start)
b.pack()
# 加载窗口，一次又一次地循环
window.mainloop()