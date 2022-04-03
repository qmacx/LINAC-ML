import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import initializers
from sklearn.metrics import precision_score, confusion_matrix, classification_report, accuracy_score
from keras import callbacks
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline


# Baseline SVM

data = pd.read_csv('../data/DxDyfirst3Quads10000.csv')
data.columns=['Label', 'Quad','Angle','CxOTR1','CyOTR1','CxOTR2','CyOTR2','CxOTR3','CyOTR3','CxOTR4','CyOTR4', 'CxOTR5','CyOTR5','CxOTR6','CyOTR6','CxOTR7','CyOTR7']
df = data.copy()

# Mapping features
dfx = data[data['Label'] == 'Dx'] 
dfy = data[data['Label'] == 'Dy']
nm = data[data['Label'] == 'No_Misalign']
print(dfx.head())
mapx = {'QM1': 'QM1_dx', 'QM2': 'QM2_dx', 'QM3': 'QM3_dx'}
mapy = {'QM1': 'QM1_dy', 'QM2': 'QM2_dy', 'QM3': 'QM3_dy'}

dfx['Quad'] = [mapx[i] for i in dfx['Quad']]
dfy['Quad'] = [mapy[i] for i in dfy['Quad']]
df = pd.concat([dfx, dfy, nm], axis=0)

# Feature selection
features = df.drop(['Label', 'Quad', 'Angle'], axis=1)
target = df['Quad']

encoder = preprocessing.LabelEncoder()
encoder.fit(target)
target = encoder.transform(target)

# scaling features
scaler = StandardScaler()
features = scaler.fit_transform(features)

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state = 50)

# model 
svm = SVC(kernel='rbf', gamma=0.5, C=0.1)
model = svm.fit(features_train, target_train)
pred = model.predict(features_test)

accuracy = accuracy_score(target_test, pred)
print('SVM RBF Accuracy {}%'.format(accuracy*100))
