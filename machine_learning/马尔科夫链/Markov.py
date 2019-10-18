import numpy as np

'''
@author zrq
@time 2019-09-25
马尔科夫链

此练习假设初始状态为t0=[0.3,0.2,0.5]t0​=[0.1,0.2,0.7]，利用马尔科夫链估算之后的状态。



'''
def markov():
    #初始状态
    init_array = np.array([0.3, 0.2, 0.5])
    #状态转移矩阵
    transfer_matrix = np.array([[0.9, 0.075, 0.025],
                               [0.15, 0.8, 0.05],
                               [0.25, 0.25, 0.5]])
    restmp = init_array
    #矩阵内积相乘
    for i in range(100):
        res = np.dot(restmp, transfer_matrix)
        print (i, "\t", res)
        restmp = res

if __name__ == '__main__':
    markov()


'''
output:
E:\anaconda3\python.exe E:/pycharmworkspace/credit/0925_p/Markov.py
0 	 [0.425  0.3075 0.2675]
1 	 [0.4955  0.34475 0.15975]
2 	 [0.5376 0.3529 0.1095]
3 	 [0.56415  0.350015 0.085835]
4 	 [0.581696 0.343782 0.074522]
5 	 [0.5937242 0.3372833 0.0689925]
6 	 [0.6021924  0.33160408 0.06620352]
7 	 [0.60826465 0.32699857 0.06473677]
8 	 [0.61267217 0.3234029  0.06392493]
9 	 [0.61589662 0.32065397 0.06344942]
10 	 [0.61826741 0.31857777 0.06315482]
11 	 [0.62001604 0.31702098 0.06296298]
12 	 [0.62130833 0.31585873 0.06283294]
13 	 [0.62226454 0.31499335 0.06274212]
14 	 [0.62297262 0.31435005 0.06267734]
15 	 [0.6234972  0.31387232 0.06263049]
16 	 [0.62388595 0.31351777 0.06259629]
17 	 [0.62417409 0.31325473 0.06257118]
18 	 [0.62438768 0.31305964 0.06255268]
19 	 [0.62454603 0.31291496 0.06253901]
20 	 [0.62466342 0.31280767 0.06252891]
21 	 [0.62475046 0.31272812 0.06252142]
22 	 [0.62481499 0.31266914 0.06251588]
23 	 [0.62486283 0.3126254  0.06251177]
24 	 [0.6248983  0.31259298 0.06250873]
25 	 [0.6249246  0.31256893 0.06250647]
26 	 [0.62494409 0.31255111 0.0625048 ]
27 	 [0.62495855 0.31253789 0.06250356]
28 	 [0.62496927 0.3125281  0.06250264]
29 	 [0.62497721 0.31252083 0.06250195]
30 	 [0.62498311 0.31251544 0.06250145]
31 	 [0.62498747 0.31251145 0.06250107]
32 	 [0.62499071 0.31250849 0.0625008 ]
33 	 [0.62499311 0.31250629 0.06250059]
34 	 [0.6249949  0.31250467 0.06250044]
35 	 [0.62499622 0.31250346 0.06250032]
36 	 [0.62499719 0.31250257 0.06250024]
37 	 [0.62499792 0.3125019  0.06250018]
38 	 [0.62499846 0.31250141 0.06250013]
39 	 [0.62499886 0.31250105 0.0625001 ]
40 	 [0.62499915 0.31250078 0.06250007]
41 	 [0.62499937 0.31250057 0.06250005]
42 	 [0.62499953 0.31250043 0.06250004]
43 	 [0.62499965 0.31250032 0.06250003]
44 	 [0.62499974 0.31250023 0.06250002]
45 	 [0.62499981 0.31250017 0.06250002]
46 	 [0.62499986 0.31250013 0.06250001]
47 	 [0.6249999  0.3125001  0.06250001]
48 	 [0.62499992 0.31250007 0.06250001]
49 	 [0.62499994 0.31250005 0.0625    ]
50 	 [0.62499996 0.31250004 0.0625    ]
51 	 [0.62499997 0.31250003 0.0625    ]
52 	 [0.62499998 0.31250002 0.0625    ]
53 	 [0.62499998 0.31250002 0.0625    ]
54 	 [0.62499999 0.31250001 0.0625    ]
55 	 [0.62499999 0.31250001 0.0625    ]
56 	 [0.62499999 0.31250001 0.0625    ]
57 	 [0.62499999 0.3125     0.0625    ]
58 	 [0.625  0.3125 0.0625]
59 	 [0.625  0.3125 0.0625]
60 	 [0.625  0.3125 0.0625]
61 	 [0.625  0.3125 0.0625]
62 	 [0.625  0.3125 0.0625]
63 	 [0.625  0.3125 0.0625]
64 	 [0.625  0.3125 0.0625]
65 	 [0.625  0.3125 0.0625]
66 	 [0.625  0.3125 0.0625]
67 	 [0.625  0.3125 0.0625]
68 	 [0.625  0.3125 0.0625]
69 	 [0.625  0.3125 0.0625]
70 	 [0.625  0.3125 0.0625]
71 	 [0.625  0.3125 0.0625]
72 	 [0.625  0.3125 0.0625]
73 	 [0.625  0.3125 0.0625]
74 	 [0.625  0.3125 0.0625]
75 	 [0.625  0.3125 0.0625]
76 	 [0.625  0.3125 0.0625]
77 	 [0.625  0.3125 0.0625]
78 	 [0.625  0.3125 0.0625]
79 	 [0.625  0.3125 0.0625]
80 	 [0.625  0.3125 0.0625]
81 	 [0.625  0.3125 0.0625]
82 	 [0.625  0.3125 0.0625]
83 	 [0.625  0.3125 0.0625]
84 	 [0.625  0.3125 0.0625]
85 	 [0.625  0.3125 0.0625]
86 	 [0.625  0.3125 0.0625]
87 	 [0.625  0.3125 0.0625]
88 	 [0.625  0.3125 0.0625]
89 	 [0.625  0.3125 0.0625]
90 	 [0.625  0.3125 0.0625]
91 	 [0.625  0.3125 0.0625]
92 	 [0.625  0.3125 0.0625]
93 	 [0.625  0.3125 0.0625]
94 	 [0.625  0.3125 0.0625]
95 	 [0.625  0.3125 0.0625]
96 	 [0.625  0.3125 0.0625]
97 	 [0.625  0.3125 0.0625]
98 	 [0.625  0.3125 0.0625]
99 	 [0.625  0.3125 0.0625]

Process finished with exit code 0



'''