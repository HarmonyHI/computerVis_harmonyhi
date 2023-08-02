import sys
import time
# 调用自己创建的类
import svm
import os
import joblib

# 模型位置的获取
path = sys.path[0]
model_path = os.path.join(path, r'svm.model')
# 测试集数据加载
path = sys.path[0]
tbasePath = os.path.join(path, r"img_test")
tst = time.clock()
# 加载模型
clf = joblib.load(model_path)
testpath = tbasePath

# 读取所有图片
tflist = svm.get_file_list(testpath)
# 数据转化为图片矩阵和标签
tdataMat, tdataLabel = svm.read_and_convert(tflist)
print("测试集数据维度为：{0}, 标签数量：{1}".format(tdataMat.shape, len(tdataLabel)))

# 效果预测
score_st = time.clock()
score = clf.score(tdataMat, tdataLabel)
score_et = time.clock()
print("计算准确率花费{:.6f}秒.".format(score_et - score_st))
print("准确率：{:.6f}".format(score))
print("错误率：{:.6f}".format((1-score)))
tet = time.clock()
print("测试总耗时{:.6f}秒.".format(tet - tst))







