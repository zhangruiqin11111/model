import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def plot_feature_importances(feature_importances, title, feature_names):
    # Normalize the importance values
    feature_importances = 100.0 * (feature_importances / max(feature_importances))

    # Sort the values and flip them
    index_sorted = np.flipud(np.argsort(feature_importances))

    # Arrange the X ticks
    pos = np.arange(index_sorted.shape[0]) + 0.5

    # Plot the bar graph
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()

if __name__=='__main__':
    # Load housing data
    '''
    #数据集下载地址：https://archive.ics.uci.edu/ml/datasets/Housing不过scikit-learn提供了数据接口，可以直接通过下面的
    代码加载数据：
    '''
    housing_data = datasets.load_boston()

    # Shuffle the data把输入数据与输出结果分成不同的变量。我们可以通过shuffle函数把数据的顺序打乱：
    # 参数random_state用来控制如何打乱数据，让我们可以重新生成结果。
    X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

    # Split the data 80/20 (80% for training, 20% for testing)
    '''
    接下来把数据分成训练数据集和测试数据集，其中80%的数据用于训练，剩余20%的数据用于测试：
    
    '''
    num_training = int(0.8 * len(X))
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]
    '''
    拟合一个决策树回归模型,选一个最大深度为4的决策树，这样可以限制决策树不变成任意深度,训练数据
    '''
    # Fit decision tree regression model
    dt_regressor = DecisionTreeRegressor(max_depth=4)
    dt_regressor.fit(X_train, y_train)

    # Fit decision tree regression model with AdaBoost 再用带AdaBoost算法的决策树回归模型进行拟合：
    ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
    ab_regressor.fit(X_train, y_train)

    # Evaluate performance of Decision Tree regressor评价决策树回归器的训练效果：
    y_pred_dt = dt_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_dt)
    evs = explained_variance_score(y_test, y_pred_dt)
    print ("\n#### Decision Tree performance ####")
    print ("Mean squared error =", round(mse, 2))
    print ("Explained variance score =", round(evs, 2))

    # Evaluate performance of AdaBoost
    y_pred_ab = ab_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_ab)
    evs = explained_variance_score(y_test, y_pred_ab)
    print ("\n#### AdaBoost performance ####")
    print ("Mean squared error =", round(mse, 2))
    print ("Explained variance score =", round(evs, 2))

    # Plot relative feature importances
    plot_feature_importances(dt_regressor.feature_importances_,
            'Decision Tree regressor', housing_data.feature_names)
    plot_feature_importances(ab_regressor.feature_importances_,
            'AdaBoost regressor', housing_data.feature_names)


    '''
    前面的结果表明，AdaBoost算法可以让误差更小，且解释方差分更接近1。
    
    '''