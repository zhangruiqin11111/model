import datetime

import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

from Classic_example_of_machine_learning.part08.convert_to_timeseries import convert_data_to_timeseries
'''

    数据每行包括3个值：一个年份、一个月份和一个浮点型数据
'''
# 从输入文件加载数据
input_file = 'data_hmm.txt'
data = np.loadtxt(input_file, delimiter=',')

# 将数据按照列的方向堆叠起来用于分析。我们并不需要在技术上做列堆叠，因为只有一个列，但如果你有多于一个列要进行分析，那么可以用下面的代码实现：
#  排列训练数据
X = np.column_stack([data[:,2]])

# 用4个成分创建并训练HMM。成分的个数是一个需要进行选择的超参数。这里选择4个成分，也就意味着用4个潜在状态生成数据。
print ("\nTraining HMM....")
num_components = 4
model = GaussianHMM(n_components=num_components, covariance_type="diag", n_iter=1000)
model.fit(X)

# 预测HMM的隐藏状态
hidden_states = model.predict(X)

print ("\nMeans and variances of hidden states:")
# 计算这些隐藏状态的均值和方差：
for i in range(model.n_components):
    print( "\nHidden state", i+1)
    print ("Mean =", round(model.means_[i][0], 3))
    print ("Variance =", round(np.diag(model.covars_[i])[0], 3))

#  用模型生成数据
num_samples = 500
samples, _ = model.sample(num_samples)
plt.plot(np.arange(num_samples), samples[:,0], c='black')
plt.title('Number of components = ' + str(num_components))

plt.show()
