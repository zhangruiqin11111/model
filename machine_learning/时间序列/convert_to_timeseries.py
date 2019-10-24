import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 定义一个函数来读取输入文件，该文件将序列观察结果转换为时间序列数据：
def convert_data_to_timeseries(input_file, column, verbose=False):
    # 这里将用到一个包含4列的文本文件，其中第一列表示年，第二列表示月，第三列和第四 列表示数据。
    data = np.loadtxt(input_file, delimiter=',')

    # 因为数据是按时间的前后顺序排列的，数据的第一行是起始日期，而数据的最后一行是终止日期,下面提取出数据集的起始日期和终止日期：
    start_date = str(int(data[0,0])) + '-' + str(int(data[0,1]))
    end_date = str(int(data[-1,0] + 1)) + '-' + str(int(data[-1,1] % 12 + 1))
    print(start_date,end_date)

    if verbose:
        print ("\nStart date =", start_date)
        print ("End date =", end_date)

    # 创建一个pandas变量，该变量包含了以月为间隔的日期序列
    dates = pd.date_range(start_date, end_date, freq='M')

    # 将给定的列转换为时间序列数据 可以用年和月访问这些数据（而不是索引）：
    data_timeseries = pd.Series(data[:,column], index=dates)
    # 打印出最开始的10个元素：
    if verbose:
        print ("\nTime series data:\n", data_timeseries[:10])
    # 返回时间索引变量：
    return data_timeseries

if __name__=='__main__':
    # 输入数据文件
    input_file = 'data_timeseries.txt'

    # 加载输入数据
    column_num = 2
    data_timeseries = convert_data_to_timeseries(input_file, column_num)

    # 画出数据序列数据
    data_timeseries.plot()
    plt.title('Input data')

    plt.show()