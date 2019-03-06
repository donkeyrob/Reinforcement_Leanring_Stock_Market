# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 13:31:27 2018

@author: xiang
"""

#live graph of return
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    graph_data = open('example.txt','r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(x)
            ys.append(y)
    ax1.clear()
    ax1.plot(xs, ys)

ani = animation.FuncAnimation(fig, animate, interval=100)
plt.show()