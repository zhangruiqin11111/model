import sys

import numpy as np

filename = sys.argv[1]
X = []
y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        y.append(yt)

# Train/test split
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Training data训练数据
X_train = np.array(X[:num_training]).reshape((num_training,1))
y_train = np.array(y[:num_training])

# Test data测试数据 这里用80%的数据作为训练数据集，其余20%的数据作为测试数据集。
X_test = np.array(X[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])

# Create linear regression object
from sklearn import linear_model
# 创建线性回归对象
linear_regressor = linear_model.LinearRegression()

# Train the model using the training sets 用训练数据集训练模型
linear_regressor.fit(X_train, y_train)

# Predict the output接下来用模型对测试数据集进行预测
y_test_pred = linear_regressor.predict(X_test)

# Plot outputs我们利用训练数据集训练了线性回归器。向fit方法提供输入数据即可训练模型。用下面的代码看看它如何拟合：
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.show()

# Measure performance此只选择一两个指标来评估我们的模型。通常 的做法是尽量保证均方误差最低，而且解释方差分最高
import sklearn.metrics as sm

print ("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2) )
print ("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2) )
print ("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2) )
print ("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2) )
print ("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# Model persistence
#python3中cPickle模块已经更名为_pickle,所以在python3中导入时可以使用：
import _pickle as pickle
#用程序保存模型的具体操作步骤如下。
output_model_file = '3_model_linear_regr.pkl'

with open(output_model_file, 'w') as f:
    pickle.dump(linear_regressor, f)

with open(output_model_file, 'r') as f:
    model_linregr = pickle.load(f)

y_test_pred_new = model_linregr.predict(X_test)
print( "\nNew mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2) )
