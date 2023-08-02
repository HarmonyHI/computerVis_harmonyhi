import sys
import PyQt5
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import os
import joblib
# 调用自己创建的类
import svm

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        # 设置窗口大小
        Dialog.resize(645, 475)
        # 设置打开图片按钮
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(230, 340, 141, 41))
        self.pushButton.setAutoDefault(False)
        self.pushButton.setObjectName("pushButton")
        # 设置显示标签按钮
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(220, 50, 191, 221))
        self.label.setWordWrap(False)
        self.label.setObjectName("label")
        # 设置文本编辑区域
        self.textEdit = QtWidgets.QTextEdit(Dialog)
        self.textEdit.setGeometry(QtCore.QRect(220, 280, 191, 41))
        self.textEdit.setObjectName("textEdit")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
    # 创建窗口设置
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "手写体识别"))
        self.pushButton.setText(_translate("Dialog", "打开图片"))
        self.label.setText(_translate("Dialog", "显示图片"))

class MyWindow(QMainWindow, Ui_Dialog):
    # 初始化数据
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.openImage) # 点击事件，开启下面函数

    # 点击事件函数
    def openImage(self):
        # 点击打开图片按钮时
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "img_test")
        # 获取图片宽高，显示在对话框上
        png = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(png)
        self.textEdit.setText(imgName)
        # 加载svm模型，预测选中图片的类别
        path = sys.path[0]
        model_path = os.path.join(path, r'svm.model')
        clf = joblib.load(model_path)
        dataMat=svm.img2vector(imgName)
        preResult = clf.predict(dataMat)
        # 在文本框中显示处理结果
        self.textEdit.setReadOnly(True)
        self.textEdit.setStyleSheet("color:red")
        self.textEdit.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.textEdit.setFontPointSize(9)
        self.textEdit.setText("预测的结果是：")
        self.textEdit.append(preResult[0])

# 主函数运行
if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
