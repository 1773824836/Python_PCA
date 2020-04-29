# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:15:55 2020

@author: ZLT
"""

from pca import *
from numpy import *

dataMat = replaceNanWithMean()
meanVals = mean(dataMat,axis=0)
meanRemoved = dataMat - meanVals
covMat = cov(meanRemoved,rowvar=0)
eigVals, eigVects = linalg.eig(mat(covMat))

print(shape(eigVals))
print(eigVals)

print(shape(eigVects))
print(eigVects)

import matplotlib.pyplot as plt

Var = eigVals
Var_sum = sum(Var)
Var_rate = Var/Var_sum
plt.plot(Var_rate[:20],'s-')
plt.show()


Var = eigVals
Var_sum = sum(Var)
Var_add = zeros_like(Var)
for i in range(len(Var)):
    Var_add[i] = sum(Var[:i+1])/Var_sum
plt.plot(Var_add[:20],'s-')
plt.show()


lowDMat, reconMat = pca(dataMat,6) 
print(lowDMat)
print(reconMat)

lowDMat, reconMat = pca(dataMat,20) 
print(lowDMat)
print(reconMat)
