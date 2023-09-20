import pyaudio
import wave
from keras.models import load_model
from get_mfcc_1dcnn import *
# from get_mfcc_2dcnn import *


def recording():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 8000
    RECORD_SECONDS = 2
    # RECORDID = str(self.stu_num)+'_'+str(num).zfill(2)+'_'+str(iter).zfill(2)
    WAVE_OUTPUT_FILENAME = 'voice/' + 'my.wav'
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
    print('Recorded.')



model = load_model('model/best_model xiaonew2.h5')

while True:
    recording()




    file = 'voice/' + 'my.wav'
    x_test = MFCC(file).reshape((1, 720, 1))
    result = model.predict(x_test)[0].tolist()

    j = result.index(max(result))
    print('Result: ' + str(j))


# recording()
#
#
#
#
# from keras.models import load_model
# from get_mfcc_1dcnn import *
# from get_mfcc_2dcnn import *
#
# model = load_model('model/2dcnn/2.h5')
# file='voices/' + 'my.wav'
# x_test = GetFrequencyFeature3(file).reshape((1, 225,100, 1))
# result = model.predict(x_test)[0].tolist()
#
# j = result.index(max(result))
# print('识别的结果为：'+str(j))

