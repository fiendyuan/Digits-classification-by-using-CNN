# -*- coding: utf-8 -*-

import tkinter as tk
import uuid
import os
import base64

import platform as plat

from keras import backend as K
import pyaudio
import wave
import threading
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
    global on_hit
    if on_hit == False:
        b.config(image=img_stop)
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
#     print('recorded')

'''
Main program interface drawing
'''
#Interface of GUI toolkit
window = tk.Tk()
window.title('Isolated word recognition')
window.geometry('560x450')
# Track changes in variable values
var = tk.StringVar()
# Create a new Label label control, which displays one or more lines of text and does not allow users to modify it. The relief parameter specifies the border style, and the anchor parameter controls the position of the text displayed in the Label.
label_ = tk.Label(window, textvariable=var, borderwidth=1, relief="sunken", font=12,
              width=50, height=20, anchor='nw', bg="#FFCCCC")
# Pack is a layout manager, which can be regarded as a flexible container
label_.pack()

# Define the recording button image
img_start = tk.PhotoImage(file="record.gif")
#Define recording stop button image
img_stop = tk.PhotoImage(file="stop.gif")


is_playing = False
my_thread = None
on_hit = False
tmp_file_path = 'tmp_wav'

# Text output content
string_list = []

# Determine whether the tmp_file_path path exists, and create a path if it does not exist
if not os.path.exists(tmp_file_path):
     os.mkdir(tmp_file_path)

# Create a new Button control
b = tk.Button(window, text='hit me', borderwidth=0, command=hit_me)

#Set button image
b.config(image=img_start)
b.pack()
# Load window, loop again and again
window.mainloop()