import pyaudio
import os
import wave
import numpy as np
import matplotlib.pyplot as plt
from playsound import playsound
class record():
    # def __init__(self):
    #     self.stu_num=stu_num
    #     if not(os.path.exists('voices')):
    #         os.mkdir('voices')
    #     if not(os.path.exists('waves')):
    #         os.mkdir('waves')

    def recording(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 8000
        RECORD_SECONDS = 2
        # RECORDID = str(self.stu_num)+'_'+str(num).zfill(2)+'_'+str(iter).zfill(2)2
        WAVE_OUTPUT_FILENAME = 'voices/Zhengjie/' + '7_ZHengjie_34.wav'
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


    # def visualize(self, num=None, iter=None):
    #     if num is None and iter is None:
    #         files = os.listdir('voices')
    #         for file in files:
    #             f = wave.open('voices/'+file, 'rb')
    #             params = f.getparams()
    #             nchannels, sampwidth, framerate, nframes = params[:4]
    #             strData = f.readframes(nframes)
    #             wavaData = np.frombuffer(strData, dtype=np.int16)
    #             wavaData = wavaData * 1.0 / max(abs(wavaData))
    #             wavaData = np.reshape(wavaData, [nframes, nchannels]).T
    #             f.close()
    #             time = np.arange(0, nframes) * (1.0 / framerate)
    #             time = np.reshape(time, [nframes, 1]).T
    #             plt.plot(time[0, :nframes], wavaData[0, :nframes], c="b")
    #             plt.xlabel("time(seconds)")
    #             plt.ylabel("amplitude")
    #             plt.title("Original wave_{}".format(file[:-4]))
    #             plt.savefig('waves/{}.jpg'.format(file))
    #             plt.show()
    #     else:
    #         file = '{}_{}_{}.dat'.format(str(self.stu_num), str(num).zfill(2), str(iter).zfill(2))
    #         f = wave.open('voices/' + file, 'rb')
    #         params = f.getparams()
    #         nchannels, sampwidth, framerate, nframes = params[:4]
    #         strData = f.readframes(nframes)
    #         wavaData = np.frombuffer(strData, dtype=np.int16)
    #         wavaData = wavaData * 1.0 / max(abs(wavaData))
    #         wavaData = np.reshape(wavaData, [nframes, nchannels]).T
    #         f.close()
    #         time = np.arange(0, nframes) * (1.0 / framerate)
    #         time = np.reshape(time, [nframes, 1]).T
    #         plt.plot(time[0, :nframes], wavaData[0, :nframes], c="b")
    #         plt.xlabel("time(seconds)")
    #         plt.ylabel("amplitude")
    #         plt.title("Original wave_{}_{}".format(str(num).zfill(2), str(iter).zfill(2)))
    #         plt.savefig('waves/{}.jpg'.format(file))
    #         plt.show()
    #
    # def play(self, num = None, iter = None):
    #     if num is None and iter is None:
    #         files = os.listdir('voices')
    #         for file in files:
    #             playsound('voices/'+file)
    #     else:
    #         playsound('voices/{}_{}_{}.dat'.format(self.stu_num, str(num).zfill(2), str(iter).zfill(2)))
    #
    # def record_all(self, play_after_record = False):
    #     for num in range(1, 21):
    #         for iter in range(1, 21):
    #             a.recording(num=num, iter=iter)
    #             if play_after_record is True:
    #                 self.play(num, iter)
    #         input("\nWord No. " + str(num) + "is recorded. Input anything to go on recording word No. " + str(num + 1)+'\n')
    #
    # def remove_all(self):
    #     voice=os.listdir('voices')
    #     for file in voice:
    #         os.remove('voices/'+file)
    #     os.removedirs('voices')
    #     wave=os.listdir('waves')
    #     for file in wave:
    #         os.remove('waves/'+file)
    #     os.removedirs('waves')


a = record()
a.recording()