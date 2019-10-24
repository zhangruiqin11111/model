import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Classic_example_of_machine_learning.part08 import convert_to_timeseries

# 输入数据文件
input_file = 'data_timeseries.txt'

# 加载数据
column_num = 2
data_timeseries = convert_to_timeseries.convert_data_to_timeseries(input_file, column_num)

# 假定我们希望提取给定的起始年份和终止年份之间的数据,下面做如下定义
start = '2008'
end = '2015'

# 画出给定年份范围内的数据：
plt.figure()
data_timeseries[start:end].plot()
plt.title('Data from ' + start + ' to ' + end)
# 还可以在给定月份范围内切分数据
start = '2007-2'
end = '2007-11'
plt.figure()
data_timeseries[start:end].plot()
plt.title('Data from ' + start + ' to ' + end)

plt.show()