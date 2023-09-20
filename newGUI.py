import sys
import time
from PyQt5.QtCore import QObject, pyqtSignal, QEventLoop, QTimer
from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication, QTextEdit
from PyQt5.QtGui import QTextCursor
import pyaudio
import wave
from keras.models import load_model
from get_mfcc_1dcnn import *
from record_and_test import recording


'''
控制台输出定向到Qtextedit中
'''


class Stream(QObject):
    """Redirects console output to text widget."""
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))


class GenMast(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.initUI()

        # Custom output stream.
        sys.stdout = Stream(newText=self.onUpdateText)

    def onUpdateText(self, text):
        """Write console output to text widget."""
        cursor = self.process.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.process.setTextCursor(cursor)
        self.process.ensureCursorVisible()

    def closeEvent(self, event):
        """Shuts down application on close."""
        # Return stdout to defaults.
        sys.stdout = sys.__stdout__
        super().closeEvent(event)

    def initUI(self):
        """Creates UI window on launch."""
        # Button for generating the master list.
        btnGenMast = QPushButton('Run', self)
        btnGenMast.move(450, 50)
        btnGenMast.resize(100, 200)
        btnGenMast.clicked.connect(self.genMastClicked)

        # Create the text output widget.
        self.process = QTextEdit(self, readOnly=True)
        self.process.ensureCursorVisible()
        self.process.setLineWrapColumnOrWidth(500)
        self.process.setLineWrapMode(QTextEdit.FixedPixelWidth)
        self.process.setFixedWidth(400)
        self.process.setFixedHeight(200)
        self.process.move(30, 50)

        # Set window size and title, then show the window.
        self.setGeometry(300, 300, 600, 300)
        self.setWindowTitle('ASR')
        self.show()

    def printhello(self):

        print('hello')

    # def recording(self):
    #     CHUNK = 1024
    #     FORMAT = pyaudio.paInt16
    #     CHANNELS = 1
    #     RATE = 8000
    #     RECORD_SECONDS = 2
    #     WAVE_OUTPUT_FILENAME = 'voice/' + 'my.wav'
    #     p = pyaudio.PyAudio()
    #     stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    #     print('Recording:')
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
    #
    # model = load_model('model/best_model xiaonew.h5')
    #
    # while True:
    #     recording(self=None)
    #
    #     file = 'voice/' + 'my.wav'
    #     x_test = get(file).reshape((1, 1000, 1))
    #     result = model.predict(x_test)[0].tolist()
    #
    #     j = result.index(max(result))
    #     print('识别的结果为：' + str(j))



    def genMastClicked(self):
      print('Running...')
      self.printhello()
      # self.recording()
      loop = QEventLoop()
      QTimer.singleShot(1999, loop.quit)
      loop.exec_()
      print('Done.')


if __name__ == '__main__':
    # Run the application.
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    gui = GenMast()
    sys.exit(app.exec_())