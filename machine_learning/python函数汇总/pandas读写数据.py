import pandas as pd
# 读取数据
one = pd.read_csv('../One.csv')
print(one.head())

hsb2 = pd.read_table('../hsb2.txt')
print(hsb2.head())

html = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html') # Return a list
print(html)


xls = pd.read_excel('../hsb2.xlsx', sheetname=0)
print(xls.head())


# 写入文件

xls.to_csv('copyofhsb2.csv')