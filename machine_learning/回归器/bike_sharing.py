import sys
import csv

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def plot_feature_importances(feature_importances, title, feature_names):
    # Normalize the importance values 将重要性值标准化
    feature_importances = 100.0 * (feature_importances / max(feature_importances))

    # Sort the values and flip them 将得分从高到低排序
    index_sorted = np.flipud(np.argsort(feature_importances))

    # Arrange the X ticks让X坐标轴上的标签居中显示
    pos = np.arange(index_sorted.shape[0]) + 0.5

    # Plot the bar graph
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()

'''

    加入了csv程序包来读取CSV文件,我们从CSV文件读取了所有数据。把数据显示在图形中时，特征名称非常有
    用。把特征名称数据从输入数值中分离出来，并作为函数返回值
'''
def load_dataset(filename):
    file_reader = csv.reader(open(filename, 'rb'), delimiter=',')
    X, y = [], []
    for row in file_reader:
        X.append(row[2:13])
        y.append(row[-1])

    # 提取特征名称
    feature_names = np.array(X[0])

    # 将第一行特征名称移除，仅保留数值
    return np.array(X[1:]).astype(np.float32), np.array(y[1:]).astype(np.float32), feature_names

if __name__=='__main__':
    # Load the dataset from the input file
    X, y, feature_names = load_dataset(sys.argv[1])
    X, y = shuffle(X, y, random_state=7)

    # 读取数据，并打乱数据顺序，让新数据与原来文件中数据排列的顺序没有关联性：
    num_training = int(0.9 * len(X))
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    # 将90%的数据用于训练，剩余10%的数据用于测试
    '''
    参数n_estimators是指评估器（estimator）的数量，表示随机森林需要使用的决策
    树数量；参数max_depth是指每个决策树的最大深度；参数min_samples_split是指决策树分
    裂一个节点需要用到的最小数据样本量。

    '''
    rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10, min_samples_split=1)
    rf_regressor.fit(X_train, y_train)

    # 训练回归器
    #评价随机森林回归器的训练效果
    y_pred = rf_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    print ("\n#### Random Forest regressor performance ####")
    print ("Mean squared error =", round(mse, 2))
    print ("Explained variance score =", round(evs, 2))

    # Plot relative feature importances
    plot_feature_importances(rf_regressor.feature_importances_, 'Random Forest regressor', feature_names)