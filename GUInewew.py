import sys
import time
from PyQt5.QtCore import QObject, pyqtSignal, QEventLoop, QTimer, QThread
from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication, QTextEdit
from PyQt5.QtGui import QTextCursor
import pyaudio
import wave
from keras.models import load_model
from get_mfcc_1dcnn import *

class myThread(QThread):
     signalForText = pyqtSignal(str)

     def __init__(self, param=None, parent=None):
         super(MyThread, self).__init__(parent)
         # If there are parameters, they can be encapsulated in a class
         self.param = param

     def write(self, text):
         self.signalForText.emit(str(text)) # Emit signal

     def run(self):
         # Pass the command command to be executed through cmdlist[self.param]
         p = subprocess.Popen(cmdlist[self.param], stdout=subprocess.PIPE, stderr=subprocess.PIPE) # Pass parameters through member variables
         while True:
             result = p.stdout.readline()
             # print("result{}".format(result))
             if result != b'':
                 print(result.decode('utf-8').strip('\r\n')) # UTF-8 decode the result
                 self.write(result.decode('utf-8').strip('\r\n'))
             else:
                 break
         while True:
             result = p.stderr.readline()
             # print("result{}".format(result))
             if result != b'':
                 print(result.decode('utf-8').strip('\r\n')) # UTF-8 decode the result for display
                 self.write(result.decode('utf-8').strip('\r\n'))
             else:
                 break
         p.stdout.close()
         p.stderr.close()
         p.wait()
         # # Demo code
         # for i in range(5):
         # print(i)
         # # self.signalForText.emit(str(i)) # Emit signal
         # self.write(i)
         # time.sleep(1)
         # print("End")


class myWidget(QWidget):
     def __init__(self):
         super(newTab, self).__init__()
         self.initUI()

     def initUI(self):
         # Execute program button
         exec_but = QPushButton("Start execution", optionFrame)
         exec_but.move(800, 0)
         exec_but.clicked.connect(self.realtime_display) # Start the thread in the bound slot function

         self.runWidget = QWidget() # Component used to display running results
         self.runWidgetLayout = QHBoxLayout(self.runWidget)
         self.runTextBrowser = QTextBrowser() #Put the running results into the text browser
         self.runWidgetLayout.addWidget(self.runTextBrowser)
         self.runTextBrowser.ensureCursorVisible() # The cursor is available

     def realtime_display(self): # Start the thread in the bound slot function
         print('Running...')
         index = self.cmdCombobox.currentIndex()
         try:
             self.t = MyThread(index)
             # The thread signal is bound to the slot function onUpdateText responsible for writing to the text browser
             self.t.signalForText.connect(self.onUpdateText)
             self.t.start()
         except Exception as e:
             raise e

         loop = QEventLoop()
         QTimer.singleShot(2000, loop.quit)
         loop.exec_()

     def onUpdateText(self, text):
         cursor = self.runTextBrowser.textCursor()
         cursor.movePosition(QTextCursor.End)
         self.runTextBrowser.append(text)
         self.runTextBrowser.setTextCursor(cursor)
         self.runTextBrowser.ensureCursorVisible()