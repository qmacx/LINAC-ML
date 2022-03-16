import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import precision_score, confusion_matrix, classification_report, accuracy_score, mean_absolute_error
from keras import callbacks
from keras.utils.vis_utils import plot_model
from keras import backend as K

# Deep Neural Network for classification of misaligned, single components for all quadrupole elements

data = pd.read_csv('./data/data.csv')
OTRindex = np.arange(0, len(data.columns)-3)
cols = ['Quad', 'Labels', 'Angle']
OTRcols = ['OTR{}'.format(x) for x in OTRindex]
cols.extend(OTRcols)
data.columns=cols
data['OTR42'][13986] = -0.00022418886038814653

ids = np.arange(0, 44)
chicaneids = np.arange(13, 32)
chicanecols = ['OTR{}'.format(x) for x in chicaneids]
quadcols = ['OTR{}'.format(x) for x in ids if x not in chicaneids]
chicanerows = ['B1', 'B2', 'B3', 'B4']
quadrows = ['QM1', 'QM2', 'QM3', 'QM4', 'QM5', 'QM6']

data = data.drop(chicanecols, axis=1)
data = data[~data.Quad.isin(chicanerows)]
data = data.drop(['OTR42'], axis=1)
# Mapping features
dfx = data[data['Labels'] == 'DX']
dfy = data[data['Labels'] == 'DY']
print(dfx)


'''
b1 = dfx[dfx['Quad'] == 'B1'].iloc[:, 3::2]
b2 = dfx[dfx['Quad'] == 'B2'].iloc[:, 3::2]
b3 = dfx[dfx['Quad'] == 'B3'].iloc[:, 3::2]
b4 = dfx[dfx['Quad'] == 'B4'].iloc[:, 3::2]

b1y = dfy[dfy['Quad'] == 'B1'].iloc[:, 4::2]
b2y = dfy[dfy['Quad'] == 'B2'].iloc[:, 4::2]
b3y = dfy[dfy['Quad'] == 'B3'].iloc[:, 4::2]
b4y = dfy[dfy['Quad'] == 'B4'].iloc[:, 4::2]
a1x = np.linspace(-100, 100, len(b1))
a2x = np.linspace(-100, 100, len(b2))
a3x = np.linspace(-100, 100, len(b3))
a4x = np.linspace(-100, 100, len(b4))
a1y = np.linspace(-100, 100, len(b1))
a2y = np.linspace(-100, 100, len(b2))
a3y = np.linspace(-100, 100, len(b3))
a4y = np.linspace(-100, 100, len(b4))
print(b1.head(), b1y.head())

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(np.linspace(0, 12, len(b1.iloc[500])), b1.iloc[500]*1e6, c='r', linestyle='--', label='B1')
ax.plot(np.linspace(0, 12, len(b2.iloc[500])), b2.iloc[500]*1e6, c='g', linestyle='--', label='B2')
ax.plot(np.linspace(0, 12, len(b3.iloc[500])), b3.iloc[500]*1e6, c='b', linestyle='--', label='B3')
ax.plot(np.linspace(0, 12, len(b4.iloc[500])), b4.iloc[500]*1e6, c='k', linestyle='--', label='B4')
#ax.plot(s4, qm4x.iloc[0]*1e6, linestyle='--', label='QM4')
#ax.plot(s5, qm5x.iloc[0]*1e6, linestyle='--', label='QM5')
#ax.plot(s6, qm6x.iloc[0]*1e6, linestyle='--', label='QM6')
ax.set(xlabel='s [m]', ylabel='centroid in y [$\mu m$]')

ax.legend()
ax.grid()
plt.tight_layout()
plt.show()
'''
qm1x = dfx[dfx['Quad'] == 'QM1'].iloc[:, 3::2]
qm2x = dfx[dfx['Quad'] == 'QM2'].iloc[:, 3::2]
qm3x = dfx[dfx['Quad'] == 'QM3'].iloc[:, 3::2]
qm4x = dfx[dfx['Quad'] == 'QM4'].iloc[:, 3::2]
qm5x = dfx[dfx['Quad'] == 'QM5'].iloc[:, 3::2]
qm6x = dfx[dfx['Quad'] == 'QM6'].iloc[:, 3::2]

qm1y = dfy[dfy['Quad'] == 'QM1'].iloc[:, 4::2]
qm2y = dfy[dfy['Quad'] == 'QM2'].iloc[:, 4::2]
qm3y = dfy[dfy['Quad'] == 'QM3'].iloc[:, 4::2]
qm4y = dfy[dfy['Quad'] == 'QM4'].iloc[:, 4::2]
qm5y = dfy[dfy['Quad'] == 'QM5'].iloc[:, 4::2]
qm6y = dfy[dfy['Quad'] == 'QM6'].iloc[:, 4::2]
print(qm1y.head())

s1 = np.linspace(0, 12, len(qm1x.iloc[0]))
s2 = np.linspace(0, 12, len(qm2x.iloc[0]))
s3 = np.linspace(0, 12, len(qm3x.iloc[0]))
s4 = np.linspace(0, 12, len(qm4x.iloc[0]))
s5 = np.linspace(0, 12, len(qm5x.iloc[0]))
s6 = np.linspace(0, 12, len(qm6x.iloc[0]))

s1y = np.linspace(0, 12, len(qm1y.iloc[0]))
s2y = np.linspace(0, 12, len(qm2y.iloc[0]))
s3y = np.linspace(0, 12, len(qm3y.iloc[0]))
s4y = np.linspace(0, 12, len(qm4y.iloc[0]))
s5y = np.linspace(0, 12, len(qm5y.iloc[0]))
s6y = np.linspace(0, 12, len(qm6y.iloc[0]))

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(s1, qm1x.iloc[0] * 1e6, linestyle='--', label='QM1')
ax.plot(s2, qm2x.iloc[0] * 1e6, linestyle='--', label='QM2')
ax.plot(s3, qm3x.iloc[0] * 1e6, linestyle='--', label='QM3')
ax.plot(s4, qm4x.iloc[0] * 1e6, linestyle='--', label='QM4')
ax.plot(s5, qm5x.iloc[0] * 1e6, linestyle='--', label='QM5')
ax.plot(s6, qm6x.iloc[0] * 1e6, linestyle='--', label='QM6')
ax.set(xlabel='s [m]', ylabel='centroid in x [$\mu m$]')
ax.legend()
ax.grid()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(s1y, qm1y.iloc[0] * 1e6, linestyle='--', label='QM1')
ax.plot(s2y, qm2y.iloc[0] * 1e6, linestyle='--', label='QM2')
ax.plot(s3y, qm3y.iloc[0] * 1e6, linestyle='--', label='QM3')
ax.plot(s4y, qm4y.iloc[0] * 1e6, linestyle='--', label='QM4')
ax.plot(s5y, qm5y.iloc[0] * 1e6, linestyle='--', label='QM5')
ax.plot(s6y, qm6y.iloc[0] * 1e6, linestyle='--', label='QM6')
ax.set(xlabel='s [m]', ylabel='centroid in y [$\mu m$]')
ax.legend()
ax.grid()
plt.tight_layout()
plt.show()

'''
x = np.linspace(0, 5, 1000)
y = np.exp(-x)
x2 = np.arange(0, len(x))

fig, ax = plt.subplots()
ax.plot(x2, y, c='b', label='Training loss')
ax.plot(x2, y + 0.2, c='g', label='Validation loss')
ax.set(xlabel='Epochs', ylabel='Loss', title='Training and Validation loss')
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(x2, y, c='b', label='Training loss')
ax.plot(x2, y - 0.05, c='g', label='Validation loss')
ax.set(xlabel='Epochs', ylabel='Loss', title='Training and Validation loss')
ax.legend()
plt.show()
'''
