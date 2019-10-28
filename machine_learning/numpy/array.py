# 从Python列表创建数组
import numpy as np
'''
    不同于 Python 列表，NumPy 要求数组必须包含同一类型的数 据。如果类型不匹配，NumPy 将会向上转换（如果可行）
    这里整型 被转换为浮点型：
'''
a=np.array([1, 4, 2, 5, 3])
print(a)
b=np.array([3.14, 4, 2, 3])
print(b)

# 设置数组的数据类型，可以用 dtype 关键字
c=np.array([1, 2, 3, 4], dtype='float32')
print(c)

# 不同于 Python 列表，NumPy 数组可以被指定为多维的。以下是 用列表的列表初始化多维数组的一种方法
d=np.array([range(i, i + 3) for i in [2, 4, 6]])
print(d)

#  创建一个长度为10的数组，数组的值都是0
e=np.zeros(10, dtype=int)
print(e)

# # 创建一个3×5的浮点型数组，数组的值都是1
f=np.ones((3, 5), dtype=float)
print(f)

# 创建一个3×5的浮点型数组，数组的值都是3.14
g=np.full((3,5),3.14)
print(g)

# 创建一个3×5的浮点型数组，数组的值是一个线性序列 # 从0开始，到20结束，步长为2 # （它和内置的range()函数类似）
h=np.arange(0, 20, 2)
print(h)

# 创建一个5个元素的数组，这5个数均匀地分配到0~1 np.linspace(0, 1, 5)
i=np.linspace(0, 1, 5)
print(i)


import numpy as np
j=np.random.seed(0) # 设置随机数种子
# print(j)
x1 = np.random.randint(10, size=6) # 一维数组
x2 = np.random.randint(10, size=(3, 4)) # 二维数组
x3 = np.random.randint(10, size=(3, 4, 5)) # 三维数组

print("x3 ndim: ", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)