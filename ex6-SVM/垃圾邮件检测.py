"""
案例：判断一封邮件是否是垃圾邮件
"""

from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import scipy.io as sio

data1 = sio.loadmat('data/spamTrain.mat')  # training data
data2 = sio.loadmat('data/spamTest.mat') # Testing data
print(data1.keys(), data2.keys())
X, y = data1['X'], data1['y']
X_test, y_test = data2['Xtest'], data2['ytest']
print(X.shape, y.shape, X_test.shape, y_test.shape)
svc = svm.SVC()
svc.fit(X, y.flatten())
pred = svc.predict(X_test)
print(svc)
print(metrics.classification_report(y_test.flatten(), pred))

# 线性回归
logit = LogisticRegression()
logit.fit(X, y.flatten())
pred_l = logit.predict(X_test)
print(logit)
print(metrics.classification_report(y_test.flatten(), pred_l))

print(X)
