import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('./data/completed_data.csv')

angle = data['angleb1']
centroid = np.array(data['centroid'])

fig, ax = plt.subplots()
ax.set(xlabel='angle [mm rad]', ylabel='centroid [m]', title='centroid vs dipole bending angle for b1=b4')
ax.plot(angle, centroid)
plt.tight_layout()
plt.show()

print(centroid.max() - centroid.min())
