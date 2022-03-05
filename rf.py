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
from sklearn.model_selection import cross_val_score
from keras import callbacks
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from sklearn.ensemble import RandomForestClassifier 
from sklearn.pipeline import make_pipeline

# Baseline Random Forest Classifier

data = pd.read_csv('PITCHfirst3Quads10000.csv')
data = pd.read_csv('DxDyfirst3Quads10000.csv')
data.columns=['Label', 'Quad','Angle','CxOTR1','CyOTR1','CxOTR2','CyOTR2','CxOTR3','CyOTR3','CxOTR4','CyOTR4', 'CxOTR5','CyOTR5','CxOTR6','CyOTR6','CxOTR7','CyOTR7']
df = data.copy()



QM1 = df[df['Quad'] == 'QM1']
QM2 = df[df['Quad'] == 'QM2']
QM3 = df[df['Quad'] == 'QM3']
NO = df[df['Quad'] == 'No_Misalign']
print(QM1, QM2, QM3, NO)


df = df[df['Label'] == 'Dx']

# Machine Learning
features = df.drop(['Label', 'Quad', 'Angle'], axis=1)
target = df['Quad']

encoder = preprocessing.LabelEncoder()
encoder.fit(target)
target = encoder.transform(target)

# scaling features
#scaler = StandardScaler()
#features = scaler.fit_transform(features)

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state = 50)

# model
rf = RandomForestClassifier()

model = rf.fit(features_train, target_train)
pred = model.predict(features_test)

accuracy = accuracy_score(target_test, pred)
print('Random Forest Accuracy {}%'.format(accuracy*100))
scores = cross_val_score(model, features_train, target_train, cv=10)
print('Cross Val Score: ', scores)
print(scores.mean())


cm = confusion_matrix(target_test, pred)
ax = sns.heatmap(cm, annot=True)
ax.set(title='Confusion Matrix for Random Forest Classifier', xlabel='Predicted Value', ylabel='True Value')
plt.show()


print(target_test - pred)
