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

# Deep Neural Network for classification of misaligned, single components over the whole beamline

data = pd.read_csv('./data/data.csv')
OTRindex = np.arange(0, len(data.columns)-3)
cols = ['Quad', 'Labels', 'Angle']
OTRcols = ['OTR{}'.format(x) for x in OTRindex]
cols.extend(OTRcols)
data.columns=cols

# Mapping features
dfx = data[data['Labels'] == 'DX']
dfy = data[data['Labels'] == 'DY']

mapx = {'QM1': 'QM1_dx', 'QM2': 'QM2_dx', 'QM3': 'QM3_dx', 
        'B1': 'B1_dx', 'B2': 'B2_dx', 'B3': 'B3_dx', 'B4': 'B4_dx',
        'QM4': 'QM4_dx', 'QM5': 'QM5_dx', 'QM6': 'QM6_dx'}

mapy = {'QM1': 'QM1_dy', 'QM2': 'QM2_dy', 'QM3': 'QM3_dy', 
        'B1': 'B1_dy', 'B2': 'B2_dy', 'B3': 'B3_dy', 'B4': 'B4_dy',
        'QM4': 'QM4_dy', 'QM5': 'QM5_dy', 'QM6': 'QM6_dy'}

dfx['Quad'] = [mapx[i] for i in dfx['Quad']]
dfy['Quad'] = [mapy[i] for i in dfy['Quad']]
df = pd.concat([dfx, dfy], axis=0)
df.to_csv('mldata.csv')

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

print(features.min(), features.max())


# Split
X_train, X_test, Yclf_train, Yclf_test, Yreg_train, Yreg_test = train_test_split(features, clf_target, reg_target, test_size=0.33, random_state = 10)

# DNN model building
inputs = tf.keras.layers.Input(shape=(inputdims,))
hidden1 = tf.keras.layers.Dense(units=inputdims * 2/3, kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')(inputs)
hidden2 = tf.keras.layers.Dense(units=inputdims * 2/3, kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')(hidden1)
hidden3 = tf.keras.layers.Dense(units=inputdims * 2/3, kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')(hidden2)
#dropout = tf.keras.layers.Dropout(0.05)(hidden2, training=True)
clf_outputs = tf.keras.layers.Dense(units=outputdims, kernel_initializer=tf.keras.initializers.glorot_normal(), activation='softmax')(hidden3)
reg_outputs = tf.keras.layers.Dense(units=1, activation='linear')(hidden3)
dnn = tf.keras.models.Model(inputs=inputs, outputs=[clf_outputs, reg_outputs])

# metrics functions

def model_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


# compile, train, predict
stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.001, patience=20, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True)

opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
dnn.compile(loss=['categorical_crossentropy', 'mse'], metrics=[model_acc], optimizer=opt)
history = dnn.fit(X_train, [Yclf_train, Yreg_train], batch_size=100, epochs=2000, validation_split=0.33)
clf_pred, reg_pred = dnn.predict(X_test)
clf_pred = np.argmax(clf_pred, axis=1)


# results
mapping = {'B1_dx': 0, 'B1_dy': 1, 'B2_dx': 2, 'B2_dy': 3, 'B3_dx': 4, 'B3_dy': 5, 'B4_dx': 6, 'B4_dy': 7,
'QM1_dx': 8, 'QM1_dy': 9, 'QM2_dx': 10, 'QM2_dy': 11, 'QM3_dx': 12, 'QM3_dy': 13,
'QM4_dx': 14, 'QM4_dy': 15, 'QM5_dx': 16, 'QM5_dy': 17, 'QM6_dx': 18, 'QM6_dy': 19}

keys = mapping.keys()
xlabels = list(keys)
print(xlabels)
Yclf_test = Yclf_test.idxmax(axis=1)
Yclf_test = [mapping[i] for i in Yclf_test] # replaces strings with mapped value
clf_acc = accuracy_score(Yclf_test, clf_pred)

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
ax.set(xlabel='predicted value [arb]', ylabel='true value [arb]')
plt.show()

# confusion matrix
ax = sns.heatmap(cm, xticklabels=xlabels, yticklabels=xlabels, annot=True)
ax.set(title='Confusion Matrix for Multiclass DNN prediction', xlabel='Predicted Value', ylabel='True Value')
plt.show()

# uncertainty
clf_mean, clf_std = np.mean(np.array(clf_pred), axis=0), np.std(np.array(clf_pred), axis=0)
reg_mean, reg_std = np.mean(np.array(reg_pred), axis=0), np.std(np.array(reg_pred), axis=0)

print(clf_mean, clf_std)
print(reg_mean, reg_std)
