
'''
    map/reduce: map将传入的函数依次作用到序列的每个元素，并把结果作为新的list返回；
                reduce把一个函数作用在一个序列[x1, x2, x3...]上，这个函数必须接收两个参数，
                reduce把结果继续和序列的下一个元素做累积计算

'''
import pandas as pd
myList = [-1, 1, -1, 1, -1, 1, 1]
print(map(abs, myList))
print(myList)

from functools import reduce
def powerAdd(a, b):
    return pow(a, 2) + pow(b, 2)

print(reduce(powerAdd, myList))

'''
    filter： filter()把传入的函数依次作用于每个元素，然后根据返回值是True还是False决定保留还是丢弃该元素
'''
def is_odd(x):
    return x % 3  # 0被判断为False，其它被判断为True
print(filter(is_odd, myList))


'''
    默认排序：数字大小或字母序（针对字符串）
'''
print(sorted(myList))

'''
    返回函数: 高阶函数除了可以接受函数作为参数外，还可以把函数作为结果值返回
'''

def powAdd(x, y):
    def power(n):
        return pow(x, n) + pow(y, n)
    return power


myF=powAdd(3,4)
print(myF)

'''
    - 匿名函数：高阶函数传入函数时，不需要显式地定义函数，直接传入匿名函数更方便
'''
f = lambda x: x * x
f(4)

print(map(lambda x: x * x, myList))
# 匿名函数可以传入多个参数
print(reduce(lambda x, y: x + y, map(lambda x: x * x, myList)))
# 返回函数可以是匿名函数
def powAdd1(x, y):
    return lambda n: pow(x, n) + pow(y, n)

lamb = powAdd1(3, 4)
lamb(2)
# 可以通过安装第三方包来增加系统功能，也可以自己编写模块。引入模块有多种方式
df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
print(df.head(10))

from pandas import DataFrame
# 可以通过安装第三方包来增加系统功能，也可以自己编写模块。引入模块有多种方式
df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
print(df1.head(5))

# 尽量不使用 'from ... import *' 这种方式，可能造成命名混乱
from pandas import *

df2 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
print(crosstab(df2.key, df2.data1))