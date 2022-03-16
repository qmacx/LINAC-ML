import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import callbacks
from keras import backend as K
from keras.utils.vis_utils import plot_model
from uncertainty import dnn_uncertainty, rmse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import precision_score, confusion_matrix, classification_report, accuracy_score, mean_squared_error

data = pd.read_csv('./data/DxDyfirst3Quads10000.csv')
data = data.drop(labels=range(30000, 39997), axis=0) # duplicate entries
data.columns=['Label','Quad','Angle','CxOTR1','CyOTR1','CxOTR2','CyOTR2','CxOTR3','CyOTR3','CxOTR4','CyOTR4', 'CxOTR5','CyOTR5','CxOTR6','CyOTR6','CxOTR7','CyOTR7']

# Mapping features
dfx = data[data['Label'] == 'Dx'] 
dfy = data[data['Label'] == 'Dy']
nm = data[data['Label'] == 'No_Misalign']
mapx = {'QM1': 'QM1_dx', 'QM2': 'QM2_dx', 'QM3': 'QM3_dx'}
mapy = {'QM1': 'QM1_dy', 'QM2': 'QM2_dy', 'QM3': 'QM3_dy'}

dfx['Quad'] = [mapx[i] for i in dfx['Quad']]
dfy['Quad'] = [mapy[i] for i in dfy['Quad']]
df = pd.concat([dfx, dfy, nm], axis=0)

# Feature selection and normalisation
scaler = StandardScaler()
reg_target = df['Angle']
reg_target = scaler.fit_transform(np.array(reg_target).reshape(-1, 1))
features = df.drop(['Label', 'Quad', 'Angle'], axis=1)
inputdims = len(features.columns)
features = scaler.fit_transform(features)
clf_target = pd.get_dummies(df['Quad'])
outputdims = len(clf_target.columns)

# Split
X_train, X_test, Yclf_train, Yclf_test, Yreg_train, Yreg_test = train_test_split(features, clf_target, reg_target, test_size=0.20, random_state = 10)

# DNN model building
inputs = tf.keras.layers.Input(shape=(inputdims,))
hidden1 = tf.keras.layers.Dense(units=8, kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')(inputs)
hidden2 = tf.keras.layers.Dense(units=8, kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')(hidden1)
clf_outputs = tf.keras.layers.Dense(units=outputdims, kernel_initializer=tf.keras.initializers.glorot_normal(), activation='softmax')(hidden2)
reg_outputs = tf.keras.layers.Dense(units=1, kernel_initializer=tf.keras.initializers.glorot_normal(), activation='linear')(hidden2)
dnn = tf.keras.models.Model(inputs=inputs, outputs=[clf_outputs, reg_outputs])

# compile, train, predict
stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.001, patience=20, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True)

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
dnn.compile(loss=['categorical_crossentropy', 'mse'], optimizer=opt)
history = dnn.fit(X_train, [Yclf_train, Yreg_train], batch_size=32, epochs=5, validation_split=0.20, callbacks=[stop], verbose=0)

# results
mapping = {'No_Misalign': 0, 'QM1_dx': 1, 'QM1_dy': 2, 'QM2_dx': 3, 'QM2_dy': 4, 'QM3_dx': 5, 'QM3_dy': 6}
Yclf_test = Yclf_test.idxmax(axis=1)
Yclf_test = [mapping[i] for i in Yclf_test]
clf_pred, reg_pred = dnn.predict(X_test)
clf_pred = np.argmax(clf_pred, axis=1)

# averaging results over n=1000
clf_acc, _ = dnn_uncertainty(X_test, Yclf_test, dnn, 1000)
reg_acc, _ = rmse(X_test, Yclf_test, dnn, 1000)
print('PMQ Regression RMSE: {:.4f}'.format(reg_acc))
print('PMQ Classifier Accuracy: {:.4f}%'.format(clf_acc*100))

# plots
history_df = pd.DataFrame(history.history)
plt.plot(history_df.loc[:, ['loss']], color='blue', label='Training loss')
plt.plot(history_df.loc[:, ['val_loss']], color='green', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="best")
plt.savefig('./plots/pmq_val.png')

fig, ax = plt.subplots()
ax.scatter(reg_pred, Yreg_test, c='k', s=0.5, marker='x')
ax.set(xlabel='predicted value[arb]', ylabel='true value [arb]')
plt.savefig('./plots/pmq_reg.png')

# confusion matrix
cm = confusion_matrix(Yclf_test, clf_pred)
xlabels = ['None', 'QM1x', 'QM1y', 'QM2x', 'QM2y', 'QM3x', 'QM3y'] 
ax = sns.heatmap(cm, xticklabels=xlabels, yticklabels=xlabels, annot=True)
ax.set(xlabel='Predicted Value', ylabel='True Value')
plt.savefig('./plots/pmq_conf.png')
