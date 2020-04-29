# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:58:08 2020

@author: ZLT
"""

import pca
import matplotlib
import matplotlib.pyplot as plt
from numpy import *

dataMat = pca.loadDataSet('testSet.txt')
print(dataMat)
lowDMat, reconMat = pca.pca(dataMat,1) #变一维
#lowDMat, reconMat = pca.pca(dataMat,2) #变二维

print(shape(lowDMat))
print(shape(reconMat))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0],marker='^',s=90)
ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')


plt.show()

dataMat = pca.loadDataSet('testSet3.txt')
print(dataMat)

from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax1 = Axes3D(fig)
ax1.scatter3D(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0],dataMat[:,2].flatten().A[0], cmap='Blues')  #绘制散点图
plt.show()


lowDMat2, reconMat2 = pca.pca(dataMat,2) #变二维
lowDMat, reconMat = pca.pca(lowDMat2,1) #变一维
print(shape(lowDMat))
print(shape(lowDMat2))
print(shape(reconMat))
print(shape(reconMat2))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(lowDMat2[:,0].flatten().A[0],lowDMat2[:,1].flatten().A[0],marker='^',s=90)
ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')

plt.show()



print(reconMat)





















