# -*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt #导入图像库

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

dish_profit = 'catering_dish_profit.xls' #餐饮菜品盈利数据
data = pd.read_excel(dish_profit, index_col = u'菜品名')
data = data[u'盈利'].copy() # 将单独的一列复制出来，包括索引
data.sort_values(ascending = False) # 对数据倒叙
# data.sort_values(by=u'菜品名',ascending = False)

data.sort_values(ascending = False)
plt.figure(figsize=(10,10))

plt.figure(figsize = (10,10))
data.plot(kind='bar') #对data绘制条形图
plt.ylabel(u'盈利（元）')
p = 1.0*data.cumsum()/data.sum() #cumsum-->依次给出前1，2...n个数的和
p.plot(color = 'r', secondary_y = True, style = '-o',linewidth = 2) #
plt.annotate(format(p[6], '.4%'), xy = (6, p[6]), xytext=(6*0.9, p[6]*0.9),
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
#添加注释，即85%处的标记。这里包括了指定箭头样式。
plt.ylabel(u'盈利（比例）')
plt.show()