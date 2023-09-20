import librosa
import matplotlib.pyplot as plt
import librosa.display
# from get_mfcc_1dcnn import *
from scipy import signal
from scipy.fftpack import fft, fftshift
# https://huailiang.github.io/blog/2019/sound/

audio_path = 'voice/voices/1_my_27.wav'
x , sr = librosa.load(audio_path, sr=None)
print(type(x), type(sr))
# <class 'numpy.ndarray'> <class 'int'>
print(x.shape, sr)
# wave
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
plt.savefig("wave.png")
# window = signal.hamming(31)
# plt.plot(window)
# plt.title("Hamming window")
# plt.ylabel("Amplitude")
# plt.xlabel("Time (Sample)")
# plt.savefig("Hamming.png")
# plt.figure()
# A = fft(window, 2048) / (len(window)/2.0)
# freq = np.linspace(-0.5, 0.5, len(A))
# response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
# plt.plot(freq, response)
# plt.axis([-0.55, 0.55, -110, 0])
# plt.title(" ")
# plt.ylabel("Amplitude [dB]")
# plt.xlabel("Frequency [cycles per sample]")
# plt.savefig("HammingFrequency.png")
# mfcc spectrogram
# mfccs = librosa.feature.mfcc(x, sr=sr)
# print(mfccs.shape)
# librosa.display.specshow(mfccs, sr=sr, x_axis='time')
# plt.savefig("mfcc.png")
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.savefig("mfcc.png")
