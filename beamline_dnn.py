import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import precision_score, confusion_matrix, classification_report, accuracy_score, mean_squared_error
from keras import callbacks
from keras.utils.vis_utils import plot_model
from keras import backend as K
from uncertainty import probability, dnn_uncertainty, rmse

# Deep Neural Network for classification of misaligned, single components over the whole beamline

data = pd.read_csv('./data/data.csv')
OTRindex = np.arange(0, len(data.columns)-3)
cols = ['Quad', 'Labels', 'Angle']
OTRcols = ['OTR{}'.format(x) for x in OTRindex]
cols.extend(OTRcols)
data.columns=cols
data['OTR42'][13986] = -0.00022418886038814653

# Mapping features
dfx = data[data['Labels'] == 'DX']
dfy = data[data['Labels'] == 'DY']

mapx = {'QM1': 'QM1_dx', 'QM2': 'QM2_dx', 'QM3': 'QM3_dx', 
        'QM4': 'QM4_dx', 'QM5': 'QM5_dx', 'QM6': 'QM6_dx',
        'B1': 'B1_dx', 'B2': 'B2_dx', 'B3': 'B3_dx', 'B4': 'B4_dx'}

mapy = {'QM1': 'QM1_dy', 'QM2': 'QM2_dy', 'QM3': 'QM3_dy', 
        'B1': 'B1_dy', 'B2': 'B2_dy', 'B3': 'B3_dy', 'B4': 'B4_dy',
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
clf_target.to_csv('clf_target.csv')

# Split
X_train, X_test, Yclf_train, Yclf_test, Yreg_train, Yreg_test = train_test_split(features, clf_target, reg_target, test_size=0.20, random_state = 10)

# DNN model building
inputs = tf.keras.layers.Input(shape=(inputdims,))
hidden1 = tf.keras.layers.Dense(units=inputdims * 2/3, kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')(inputs)
hidden2 = tf.keras.layers.Dense(units=inputdims * 2/3, kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')(hidden1)
hidden3 = tf.keras.layers.Dense(units=inputdims * 2/3, kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')(hidden2)
clf_outputs = tf.keras.layers.Dense(units=outputdims, kernel_initializer=tf.keras.initializers.glorot_normal(), activation='softmax')(hidden3)
reg_outputs = tf.keras.layers.Dense(units=1, activation='linear')(hidden3)
dnn = tf.keras.models.Model(inputs=inputs, outputs=[clf_outputs, reg_outputs])

# compile, train, predict
stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.001, patience=20, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True)

opt = tf.keras.optimizers.SGD(learning_rate=1e-4)
dnn.compile(loss=['categorical_crossentropy', 'mse'], optimizer=opt)
history = dnn.fit(X_train, [Yclf_train, Yreg_train], batch_size=100, epochs=1000, validation_split=0.20, callbacks=[stop], verbose=0)

clf_pred, reg_pred = dnn.predict(X_test)
clf_pred = np.argmax(clf_pred, axis=1)

# results
mapping = {'B1_dx': 0, 'B1_dy': 1, 'B2_dx': 2, 'B2_dy': 3, 'B3_dx': 4, 'B3_dy': 5, 'B4_dx': 4,  'B4_dy': 5,
'QM1_dx': 6, 'QM1_dy': 7, 'QM2_dx': 8, 'QM2_dy': 9, 'QM3_dx': 10, 'QM3_dy': 11,
'QM4_dx': 12, 'QM4_dy': 13, 'QM5_dx': 14, 'QM5_dy': 15, 'QM6_dx': 16, 'QM6_dy': 17}

keys = mapping.keys()
xlabels = list(keys)
Yclf_test = Yclf_test.idxmax(axis=1)
Yclf_test = [mapping[i] for i in Yclf_test] # replaces strings with mapped value
clf_acc = accuracy_score(Yclf_test, clf_pred)

# averaging results over n=1000
reg_acc, _ = rmse(X_test, Yreg_test, dnn, 1000)
clf_acc, _ = dnn_uncertainty(X_test, Yclf_test, dnn, 1000)
cm = confusion_matrix(Yclf_test, clf_pred)

print('Beamline Regression RMSE: {:.4f}'.format(reg_acc))
print('Beamline Classifier Accuracy: {:.4f}%'.format(clf_acc*100))

# plots
history_df = pd.DataFrame(history.history)
plt.plot(history_df.loc[:, ['loss']], color='blue', label='Training loss')
plt.plot(history_df.loc[:, ['val_loss']], color='green', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="best")
plt.savefig('./plots/beamline_val.png')

fig, ax = plt.subplots()
ax.scatter(reg_pred, Yreg_test, c='k', s=0.5, marker='x')
ax.set(xlabel='predicted value [arb]', ylabel='true value [arb]')
plt.savefig('./plots/beamline_reg.png')

# confusion matrix
ax = sns.heatmap(cm, xticklabels=xlabels, yticklabels=xlabels, annot=True)
ax.set(xlabel='Predicted Value', ylabel='True Value')
plt.savefig('./plots/beamline_conf.png')
