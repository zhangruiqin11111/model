import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

'''
Gamma分布
Probability density function at x of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x
        method:
          概率密度函数:  
         pdf(x, a, loc=0, scale=1)
'''
fig=plt.figure(figsize=(18,6))#确定绘图区域尺寸
ax1=fig.add_subplot(1,2,1)#将绘图区域分成左右两块
ax2=fig.add_subplot(1,2,2)
x=np.arange(0.01,15,0.01)#生成数列

z1=st.gamma.pdf(x,0.9,scale=2)#gamma(0.9,2)密度函数对应值
z2=st.gamma.pdf(x,1,scale=2)
z3=st.gamma.pdf(x,2,scale=2)
ax1.plot(x,z1,label="a<1")
ax1.plot(x,z2,label="a=1")
ax1.plot(x,z3,label="a>1")
ax1.legend(loc='best')
ax1.set_xlabel('x')
ax1.set_ylabel('p(x)')
ax1.set_title("Gamma Distribution lamda=2")

y1=st.gamma.pdf(x,1.5,scale=2)#gamma(1.5,2)密度函数对应值
y2=st.gamma.pdf(x,2,scale=2)
y3=st.gamma.pdf(x,2.5,scale=2)
y4=st.gamma.pdf(x,3,scale=2)
ax2.plot(x,y1,label="a=1.5")
ax2.plot(x,y2,label="a=2")
ax2.plot(x,y3,label="a=2.5")
ax2.plot(x,y4,label="a=3")
ax2.set_xlabel('x')
ax2.set_ylabel('p(x)')
ax2.set_title("Gamma Distribution lamda=2")
ax2.legend(loc="best")


plt.show()


print(st.gamma.pdf.__doc__)
