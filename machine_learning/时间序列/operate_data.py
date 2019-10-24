import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Classic_example_of_machine_learning.part08.convert_to_timeseries import convert_data_to_timeseries

# 输入数据文件
input_file = 'data_timeseries.txt'

# 将用到第三列和第四列数据：
data1 = convert_data_to_timeseries(input_file, 2)
data2 = convert_data_to_timeseries(input_file, 3)
# 将数据转化为pandas的数据帧：
dataframe = pd.DataFrame({'first': data1, 'second': data2})

# 画图画出给定年份范围内的数据：
dataframe['1952':'1955'].plot()
plt.title('Data overlapped on top of each other')

#假定我们希望画出在给定年份范围内刚才加载的两列数据的不同，可以用以下方式实现：
plt.figure()
difference = dataframe['1952':'1955']['first'] - dataframe['1952':'1955']['second']
difference.plot()
plt.title('Difference (first - second)')


# 如果希望对第一列和第二列用不同的条件来过滤数据，可以指定这些条件并将其画出：
# 当“first”大于某个阈值且“second”小于某个阈值时
dataframe[(dataframe['first'] > 60) & (dataframe['second'] < 20)].plot()
plt.title('first > 60 and second < 20')

plt.show()