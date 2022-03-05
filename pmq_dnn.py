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

# Deep Neural Network for classification of misaligned, single components in PMQ stage

data = pd.read_csv('./data/DxDyfirst3Quads10000.csv')
data = data.drop(labels=range(30000, 39997), axis=0)
data.columns=['Label','Quad','Angle','CxOTR1','CyOTR1','CxOTR2','CyOTR2','CxOTR3','CyOTR3','CxOTR4','CyOTR4', 'CxOTR5','CyOTR5','CxOTR6','CyOTR6','CxOTR7','CyOTR7']




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

# Feature selection and normalisation
scaler = StandardScaler()
reg_target = df['Angle']
reg_target = scaler.fit_transform(np.array(reg_target).reshape(-1, 1))
features = df.drop(['Label', 'Quad', 'Angle'], axis=1)
features = features.drop(['CxOTR2', 'CyOTR2', 'CxOTR4', 'CyOTR4', 'CxOTR6', 'CyOTR6'], axis=1)
inputdims = len(features.columns)
features = scaler.fit_transform(features)
clf_target = pd.get_dummies(df['Quad'])
outputdims = len(clf_target.columns)

# Split
X_train, X_test, Yclf_train, Yclf_test, Yreg_train, Yreg_test = train_test_split(features, clf_target, reg_target, test_size=0.33, random_state = 10)

# DNN model building
inputs = tf.keras.layers.Input(shape=(inputdims,))
hidden1 = tf.keras.layers.Dense(units=8, kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')(inputs)
#dropout = tf.keras.layers.Dropout(0.05)(hidden1, training=True)
hidden2 = tf.keras.layers.Dense(units=8, kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')(hidden1)
clf_outputs = tf.keras.layers.Dense(units=outputdims, kernel_initializer=tf.keras.initializers.glorot_normal(), activation='softmax')(hidden2)
reg_outputs = tf.keras.layers.Dense(units=1, activation='linear')(hidden2)
dnn = tf.keras.models.Model(inputs=inputs, outputs=[clf_outputs, reg_outputs])

# metrics functions

def model_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

# compile, train, predict
stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.001, patience=20, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True)

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
dnn.compile(loss=['categorical_crossentropy', 'mse'], metrics=[model_acc], optimizer=opt)
history = dnn.fit(X_train, [Yclf_train, Yreg_train], batch_size=32, epochs=500, validation_split=0.33, callbacks=[stop])
clf_pred, reg_pred = dnn.predict(X_test)

# results
mapping = {'No_Misalign': 0, 'QM1_dx': 1, 'QM1_dy': 2, 'QM2_dx': 3, 'QM2_dy': 4, 'QM3_dx': 5, 'QM3_dy': 6}
Yclf_test = Yclf_test.idxmax(axis=1)
Yclf_test = [mapping[i] for i in Yclf_test] # replaces strings with mapped value
clf_pred = np.argmax(clf_pred, axis=1)
clf_acc = accuracy_score(Yclf_test, clf_pred)
print(clf_pred)
reg_error = mean_absolute_error(Yreg_test, reg_pred)
cm = confusion_matrix(Yclf_test, clf_pred)
print('Regression MAE: {:.4f}%'.format(reg_error*100))
print('Classifier Accuracy: {:.4f}%'.format(clf_acc*100))

# plots
history_df = pd.DataFrame(history.history)
plt.plot(history_df.loc[:, ['loss']], color='blue', label='Training loss')
plt.plot(history_df.loc[:, ['val_loss']], color='green', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="best")
plt.show()

fig, ax = plt.subplots()
ax.scatter(reg_pred[::20], Yreg_test[::20], c='k', s=0.5, marker='x')
ax.set(xlabel='predicted value[arb]', ylabel='true value [arb]')
plt.show()

# confusion matrix
xlabels = ['None', 'QM1x', 'QM1y', 'QM2x', 'QM2y', 'QM3x', 'QM3y'] 
ax = sns.heatmap(cm, xticklabels=xlabels, yticklabels=xlabels, annot=True)
ax.set(xlabel='Predicted Value', ylabel='True Value')
plt.show()

# uncertainty
clf_mean, clf_std = np.mean(np.array(clf_pred), axis=0), np.std(np.array(clf_pred), axis=0)
reg_mean, reg_std = np.mean(np.array(reg_pred), axis=0), np.std(np.array(reg_pred), axis=0)

print(clf_mean, clf_std)
print(reg_mean, reg_std)
