import numpy as np
import pandas as pd
import seaborn as sns 
import tensorflow as tf
import matplotlib.pyplot as plt 
from sklearn.metrics import precision_score, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras import callbacks
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error
from uncertainty import rmse


# Baseline Linear Regression (PMQ)


data = pd.read_csv('../data/data.csv')
OTRindex = np.arange(0, len(data.columns)-3)
cols = ['Quad', 'Labels', 'Angle']
OTRcols = ['OTR{}'.format(x) for x in OTRindex]
cols.extend(OTRcols)
data.columns=cols
data['OTR42'][13986] = -0.00022418886038814653 # rogue data point in csv
print(data.head())

# removing chicane data
chicaneids = np.arange(14, 34) 
chicanecols = ['OTR{}'.format(x) for x in chicaneids]
chicanerows = ['B1', 'B2', 'B3', 'B4']

dropids = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 34, 35, 36, 37, 38, 39] # determined by detectors.py
dropotrs = ['OTR{}'.format(x) for x in dropids]
data = data.drop(chicanecols, axis=1)
data = data[~data.Quad.isin(chicanerows)]
data = data.drop(dropotrs, axis=1)
print(data.head())

# Mapping features
dfx = data[data['Labels'] == 'DX'] 
dfy = data[data['Labels'] == 'DY']
nm = data[data['Labels'] == 'No_Misalign']

mapx = {'QM1': 'QM1_dx', 'QM2': 'QM2_dx', 'QM3': 'QM3_dx', 
        'QM4': 'QM4_dx', 'QM5': 'QM5_dx', 'QM6': 'QM6_dx'}

mapy = {'QM1': 'QM1_dy', 'QM2': 'QM2_dy', 'QM3': 'QM3_dy', 
        'QM4': 'QM4_dy', 'QM5': 'QM5_dy', 'QM6': 'QM6_dy'}


dfx['Quad'] = [mapx[i] for i in dfx['Quad']]
dfy['Quad'] = [mapy[i] for i in dfy['Quad']]
df = pd.concat([dfx, dfy, nm], axis=0)

print(df.head())


# Feature selection
features = df.drop(['Labels', 'Quad', 'Angle'], axis=1)
quads = pd.get_dummies(df['Quad'])
features = pd.concat([features, quads], axis=1)
target = data['Angle']

# Feature normalisation
scaler = StandardScaler()
features = scaler.fit_transform(features)
target = scaler.fit_transform(np.array(target).reshape(-1, 1))

# Split
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state = 10)

# Linear Regression
LR = LinearRegression()
LR.fit(features_train, target_train)

predict = LR.predict(features_test)
score = mean_squared_error(target_test, predict)
print('score: ', score)

lr_error, lr_std = rmse(features_test, target_test, LR, 100)
print('LR RMSE: ', lr_error)


fig, ax = plt.subplots()
ax.scatter(target_test, predict, c='k', s=0.5, marker='x')
ax.set(xlabel='True Value [arb]', ylabel='Predicted Value [arb]', title='Linear Regression Prediction')
plt.show()
print(target_test, predict)
