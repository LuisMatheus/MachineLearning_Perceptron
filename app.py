import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from activation_functions import BinaryStep
from perceptron import Perceptron

dataset = pd.read_csv('database/and.csv')
X = dataset.iloc[ : , 0:2].values
d = dataset.iloc[ : , 2: ].values

p = Perceptron(X, d, 0.1, BinaryStep)

p.train()

xplot = np.arange(-2,3,0.5)
yplot = list(map(lambda x: -1 * (p.W[0]/p.W[1]) * x + (p.theta/p.W[1]),xplot))


plt.plot(xplot,yplot)

for i in range(len(d)):
    if d[i] == 1:
        plt.plot(X[i,0],X[i,1],'go')
    else:
        plt.plot(X[i,0],X[i,1],'ro')