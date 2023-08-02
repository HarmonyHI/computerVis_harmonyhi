from PIL import Image
import os
import sys
import numpy as np
import time
from sklearn import svm
import joblib
import warnings
warnings.filterwarnings('ignore')

# 获取指定路径下的所有 .jpg文件
def get_file_list(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]

# 解析出 .jpg图片文件的名称
def get_img_name_str(imgPath):
    return imgPath.split(os.path.sep)[-1]

# 将 16ps * 16ps 的图像数据转换成 1*256 的 numpy 向量
# 参数： imgFIle--图像名  如：0_1.png
# 返回：1 * 256 的 numpy 向量
def img2vector(imgFile):
    # print("in img2vector func--para:{}".format(imgFile))
    img = Image.open(imgFile).convert('L')
    img_arr = np.array(img, 'i') # 16px * 16px 灰度图像
    img_normalization = np.round(img_arr/255) # 对灰度值进行归一化
    img_arr2 = np.reshape(img_normalization, (1, -1)) # 1 * 256 矩阵
    return img_arr2

# 读取转换功能
# 输入图片文件
# 输出图片矩阵和标签
def read_and_convert(imgFileList):
    dataLabel = [] # 存放类标签
    dataNum = len(imgFileList)
    dataMat = np.zeros((dataNum, 256)) # dataNum * 256 的矩阵
    for i in range(dataNum):
        imgNameStr = imgFileList[i]
        imgName = get_img_name_str(imgNameStr) # 得到 数字_实例编号.jpg
        classTag = imgName.split(".")[0].split("_")[0] # 得到 类标签（数字）
        dataLabel.append(classTag)
        dataMat[i, :] = img2vector(imgNameStr)
    return dataMat, dataLabel

# 读取训练数据
def read_all_data():
    path = sys.path[0]
    train_data_path = os.path.join(path, r'img_train')
    # 调用所有图片
    flist = get_file_list(train_data_path)
    # 转化为图片矩阵和标签
    dataMat, dataLabel = read_and_convert(flist)
    return dataMat, dataLabel

# 模型创建
def create_svm(dataMat, dataLabel, path, decision='ovr'):
    clf = svm.SVC(decision_function_shape=decision)
    rf = clf.fit(dataMat, dataLabel)
    joblib.dump(rf, path) # 存储模型
    return clf

if __name__ == '__main__':
    print('正在运行模型请稍等')
    dataMat, dataLabel = read_all_data() # 调用函数，获取图片矩阵和标签
    path = sys.path[0]
    model_path = os.path.join(path, r'svm.model')
    create_svm(dataMat, dataLabel, model_path, decision='ovr')
    print('模型训练存储完成')






