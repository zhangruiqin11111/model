'''
题目描述
    在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，
    输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。


主要思路：首先选取右上角的数字，如果该数字大于target，则该列全大于target，删除该列；如果该数字小于小于target，
         则该列全小于target，删除该行。从右上角元素开始，当没到左下角元素前，不断判断右上角元素和target的关系，
         可以不断缩小查找范围。

'''

# -*- coding:utf-8 -*-
class Solution:
    # array 二维列表
    def insert2dArray(self, seq, row, col):
        # 没有使用numpy的array
        # array = [[0] * col] * row 这种方式是浅拷贝，不好用
        array = [[0 for i in range(col)] for i in range(row)]
        for i in range(row):
            for j in range(col):
                array[i][j] = seq[i * row + j]
        return array

    def Find(self, target, array):
        # 主要思路：首先选取右上角的数字，如果该数字大于target，则该列全大于target，删除该列；
        # 如果该数字小于小于target，则该列全小于target，删除该行。
        found = False
        row = len(array)
        if row:
            col = len(array[0])
        else:
            col = 0

        if row > 0 and col > 0:
            # find index of top right-hand corner
            i = 0
            j = col - 1
            # if never meets lower-left corner
            while i < row and j >= 0:
                if array[i][j] == target:
                    found = True
                    # forget break
                    break
                elif array[i][j] > target:
                    j -= 1
                elif array[i][j] < target:
                    i += 1
        return found

if __name__ == '__main__':
    answer = Solution()
    seq = [1, 2, 8, 9, 2, 4, 9, 12, 4, 7, 10, 13, 6, 8, 11, 15]
    matrix = answer.insert2dArray(seq, 4, 4)
    print(matrix)
    print(answer.Find(7, matrix))