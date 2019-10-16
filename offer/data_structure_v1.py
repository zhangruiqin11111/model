﻿'''
输入一颗二叉树的根节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。
路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。
(注意: 在返回值的list中，数组长度大的数组靠前)


'''
# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        # write code here
        result=[]
        if not root:
            return []
        if root.left==None and root.right==None and root.val==expectNumber:
            return [[root.val]]
        else:
            left = self.FindPath(self, root.left, root.val-expectNumber)
            right = self.FindPath(self, root.right, root.val - expectNumber)
            for item in right+left:
                return result.append(item+[root.val])
        return result

