import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import callbacks
from keras import backend as K
from keras.utils.vis_utils import plot_model
from uncertainty import rmse, dnn_uncertainty
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import precision_score, confusion_matrix, classification_report, accuracy_score,mean_squared_error

# Deep Neural Network for classification of misaligned, single components for all quadrupole elements

data = pd.read_csv('./data/data.csv')
OTRindex = np.arange(0, len(data.columns)-3)
cols = ['Quad', 'Labels', 'Angle']
OTRcols = ['OTR{}'.format(x) for x in OTRindex]
cols.extend(OTRcols)
data.columns=cols
data['OTR42'][13986] = -0.00022418886038814653 # rogue data point in csv

# removing chicane data
chicaneids = np.arange(14, 34)
chicanecols = ['OTR{}'.format(x) for x in chicaneids]
chicanerows = ['B1', 'B2', 'B3', 'B4']
data = data.drop(chicanecols, axis=1)
data = data[~data.Quad.isin(chicanerows)]

# Mapping features
dfx = data[data['Labels'] == 'DX']
dfy = data[data['Labels'] == 'DY']

mapx = {'QM1': 'QM1_dx', 'QM2': 'QM2_dx', 'QM3': 'QM3_dx', 
        'QM4': 'QM4_dx', 'QM5': 'QM5_dx', 'QM6': 'QM6_dx'}

mapy = {'QM1': 'QM1_dy', 'QM2': 'QM2_dy', 'QM3': 'QM3_dy', 
        'QM4': 'QM4_dy', 'QM5': 'QM5_dy', 'QM6': 'QM6_dy'}

dfx['Quad'] = [mapx[i] for i in dfx['Quad']]
dfy['Quad'] = [mapy[i] for i in dfy['Quad']]
df = pd.concat([dfx, dfy], axis=0)

# Feature selection and normalisation
scaler = StandardScaler()
reg_target = df['Angle']
reg_target = scaler.fit_transform(np.array(reg_target).reshape(-1, 1))
clf_target = pd.get_dummies(df['Quad'])
features = df.drop(['Labels', 'Quad', 'Angle'], axis=1)
inputdims = len(features.columns)
outputdims = len(clf_target.columns)
features = scaler.fit_transform(features)

# Split
X_train, X_test, Yclf_train, Yclf_test, Yreg_train, Yreg_test = train_test_split(features, clf_target, reg_target, test_size=0.20, random_state = 10)

# DNN model building
inputs = tf.keras.layers.Input(shape=(inputdims,))
hidden1 = tf.keras.layers.Dense(units=96, kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')(inputs)
hidden2 = tf.keras.layers.Dense(units=96, kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')(hidden1)
hidden3 = tf.keras.layers.Dense(units=96, kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')(hidden2)
dropout = tf.keras.layers.Dropout(0.05)(hidden3, training=True)
clf_outputs = tf.keras.layers.Dense(units=outputdims, kernel_initializer=tf.keras.initializers.glorot_normal(), activation='softmax')(dropout)
reg_outputs = tf.keras.layers.Dense(units=1, kernel_initializer=tf.keras.initializers.glorot_normal(), activation='linear')(dropout)
dnn = tf.keras.models.Model(inputs=inputs, outputs=[clf_outputs, reg_outputs])

# compile, train, predict
stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.001, patience=20, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True)

opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
dnn.compile(loss=['categorical_crossentropy', 'mse'], metrics='accuracy', optimizer=opt)
history = dnn.fit(X_train, [Yclf_train, Yreg_train], batch_size=100, epochs=1000, validation_split=0.20, callbacks=[stop], verbose=0)

# results
clf_pred, reg_pred = dnn.predict(X_test)
clf_pred = np.argmax(clf_pred, axis=1)

mapping = {'QM1_dx': 0, 'QM1_dy': 1, 'QM2_dx': 2, 'QM2_dy': 3, 'QM3_dx': 4, 'QM3_dy': 5,
'QM4_dx': 6, 'QM4_dy': 7, 'QM5_dx': 8, 'QM5_dy': 9, 'QM6_dx': 10, 'QM6_dy': 11}

keys = mapping.keys()
xlabels = list(keys)
Yclf_test = Yclf_test.idxmax(axis=1)
Yclf_test = [mapping[i] for i in Yclf_test] # replaces strings with mapped value

# averaging results over n=1000
reg_acc, reg_rmse = rmse(X_test, Yreg_test, dnn, 1000)
clf_acc, clf_std = dnn_uncertainty(X_test, Yclf_test, dnn, 1000)

print('QMS1 Regression RMSE: {:.4f}'.format(reg_acc))
print('QMS1 Classifier Accuracy: {:.4f}%'.format(clf_acc*100))

# plots
history_df = pd.DataFrame(history.history)
plt.plot(history_df.loc[:, ['loss']], color='blue', label='Training loss')
plt.plot(history_df.loc[:, ['val_loss']], color='green', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="best")
plt.savefig('./plots/qms_val.png')

# regression
fig, ax = plt.subplots()
ax.scatter(reg_pred, Yreg_test, c='k', s=0.5, marker='x')
ax.set(xlabel='predicted value[arb]', ylabel='true value [arb]')
plt.savefig('./plots/qms_reg.png')

# confusion matrix
cm = confusion_matrix(Yclf_test, clf_pred)
ax = sns.heatmap(cm, xticklabels=xlabels, yticklabels=xlabels, annot=True)
ax.set(xlabel='Predicted Value', ylabel='True Value')
plt.savefig('./plots/qms_conf.png')

plot_model(dnn, to_file='./plots/qms_model_plot.png', show_shapes=True, show_layer_names=True)
