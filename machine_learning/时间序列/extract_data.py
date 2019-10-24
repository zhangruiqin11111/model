import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Classic_example_of_machine_learning.part08.convert_to_timeseries import convert_data_to_timeseries
'''

    分析时间序列数据的主要原因之一是从中提取出有趣的统计信息。考虑数据的本质，时间序
    列分析可以提供很多信息。本节将介绍如何提取这些统计信息。
'''
# 输入数据文件
input_file = 'data_timeseries.txt'

# 加载第三列和第四列数据：
data1 = convert_data_to_timeseries(input_file, 2)
data2 = convert_data_to_timeseries(input_file, 3)
# 创建一个pandas数据结构来保存这些数据，这个数据看着比较像词典，它有对应的键和值：
dataframe = pd.DataFrame({'first': data1, 'second': data2})

# 打印最大值和最小值
print ('\nMaximum:\n', dataframe.max())
print ('\nMinimum:\n', dataframe.min())

# 打印均值 打印数据的均值或者是每行的均值：
print ('\nMean:\n', dataframe.mean())
print ('\nMean row-wise:\n', dataframe.mean(1)[:10])

# 打印滑动均值
# pd.rolling_mean(dataframe, window=24).plot()
tmp = pd.rolling(windows=24).mean()
# 打印相关性系数
print ('\nCorrelation coefficients:\n', dataframe.corr())

#  相关性系数对于理解数据的本质来说非常有用  画出滑动相关性
plt.figure()
pd.rolling_corr(dataframe['first'], dataframe['second'], window=60).plot()

plt.show()
