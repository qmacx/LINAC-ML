import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

r = 25
x = np.arange(0, 25)
y = np.arange(0, 25)

theta = np.linspace(0, np.pi * 2, 20)
x = np.cos(theta)
y = np.sin(theta)

x2 = x.copy()
y2 = y + 0.1



fig, ax = plt.subplots()
ax.scatter(x, y, color='k', label='Nominal')
ax.scatter(0, 0, color='k')
ax.scatter(x2, y2, color='r')
ax.scatter(0, 0.1, color='r', label='Deviation')
ax.legend()
ax.set(title='Centroid', xlabel='x [cm]', ylabel='y [cm]')
plt.show()
