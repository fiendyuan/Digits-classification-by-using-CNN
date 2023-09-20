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
        # 如果有参数，可以封装在类里面
        self.param = param

    def write(self, text):
        self.signalForText.emit(str(text))  # 发射信号

    def run(self):
        # 通过cmdlist[self.param]传递要执行的命令command
        p = subprocess.Popen(cmdlist[self.param], stdout=subprocess.PIPE, stderr=subprocess.PIPE) # 通过成员变量传参
        while True:
            result = p.stdout.readline()
            # print("result{}".format(result))
            if result != b'':
                print(result.decode('utf-8').strip('\r\n'))  # 对结果进行UTF-8解码以显示中文
                self.write(result.decode('utf-8').strip('\r\n'))
            else:
                break
        while True:
            result = p.stderr.readline()
            # print("result{}".format(result))
            if result != b'':
                print(result.decode('utf-8').strip('\r\n'))  # 对结果进行UTF-8解码以显示中文
                self.write(result.decode('utf-8').strip('\r\n'))
            else:
                break
        p.stdout.close()
        p.stderr.close()
        p.wait()
        # # 演示代码
        # for i in range(5):
        #     print(i)
        #     # self.signalForText.emit(str(i))  # 发射信号
        #     self.write(i)
        #     time.sleep(1)
        # print("End")


class myWidget(QWidget):
    def __init__(self):
        super(newTab, self).__init__()
        self.initUI()

    def initUI(self):
        # 执行程序按钮
        exec_but = QPushButton("开始执行", optionFrame)
        exec_but.move(800, 0)
        exec_but.clicked.connect(self.realtime_display) # 在绑定的槽函数中启动线程

        self.runWidget = QWidget() # 用于展示运行结果的组件
        self.runWidgetLayout = QHBoxLayout(self.runWidget)
        self.runTextBrowser = QTextBrowser() # 运行结果放入文本浏览器中
        self.runWidgetLayout.addWidget(self.runTextBrowser))
        self.runTextBrowser.setFont(QFont('宋体', 12))
        self.runTextBrowser.ensureCursorVisible()  # 游标可用

    def realtime_display(self): # 在绑定的槽函数中启动线程
        print('Running...')
        index = self.cmdCombobox.currentIndex()
        try:
            self.t = MyThread(index)
            # 线程信号绑定到负责写入文本浏览器的槽函数onUpdateText
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