
import librosa
import numpy as np
import wave
from scipy.fftpack import fft
def get(file):
    y, sr = librosa.load(file, sr=None)

    mfcc_data = librosa.feature.mfcc(y=y, sr=sr,window='hamming')
    X = np.zeros((50, 20), dtype=np.float)

    mfcc_data = mfcc_data.reshape((-1, 20))

    X[:len(mfcc_data)] = mfcc_data

    return X
        # .T.reshape(50, 20, 1)
'''

[20,25]
[20,10]
[20,50]
'''
b = get('voice/recordings/0_george_2.wav')
print(b)
print(b.shape)

#
# def get_tezheng(filename):
#     wav = wave.open(filename, "rb")
#     num_frame = wav.getnframes()
#
#     num_channel = wav.getnchannels()
#     framerate = wav.getframerate()
#
#     num_sample_width = wav.getsampwidth()
#     str_data = wav.readframes(num_frame)
#     wav.close()
#     wave_data = np.fromstring(str_data, dtype=np.short)
#     wave_data.shape = -1, num_channel
#     wave_data = wave_data.T
#     return wave_data, framerate
#
# # 汉明窗
# x=np.linspace(0, 200 - 1, 200, dtype = np.int64)
# w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (200 - 1) )
#
#
# def GetFrequencyFeature3(filename):
#
#     wavsignal, fs = get_tezheng(filename)
#     time_window = 25
#     window_length = fs / 1000 * time_window
#
#     wav_arr = np.array(wavsignal)
#
#     wav_length = wav_arr.shape[1]
#
#     range0_end = int(len(wavsignal[0]) / fs * 1000 - time_window) // 10
#
#     data_input = np.zeros((range0_end, 100), dtype=np.float)
#     data_line = np.zeros((1, 200), dtype=np.float)
#
#     for i in range(0, range0_end):
#         p_start = i * 80
#         p_end = p_start + 200
#         data_line = wav_arr[0, p_start:p_end]
#         data_line = data_line * w
#         data_line = np.abs(fft(data_line)) / wav_length
#         data_input[i] = data_line[0:100]
#
#     data_input = np.log(data_input + 1)
#
#     X = np.zeros((225,100), dtype=np.float)
#
#
#     X[:len(data_input)] = data_input
#
#
#     return X.reshape((225,100,1))


# import os
# list=[]
# for i in os.listdir('recordings/'):
#
#     a = GetFrequencyFeature3('recordings/'+i)
#     list.append(a.shape[0])
#     print(a.shape)
# print(max(list))




# import os
# list= []
# for i in os.listdir('recordings/'):
#     list.append(get('recordings/'+i))
#
# print(max(list))
