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


data = pd.read_csv('DxDyfirst3Quads10000.csv')
data = data.drop(labels=range(30000, 39997), axis=0)
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

# Feature selection
features = df.drop(['Label', 'Quad', 'Angle'], axis=1)
features = features.drop(['CxOTR2','CyOTR2','CxOTR4','CyOTR4','CxOTR6','CyOTR6'], axis=1)
quads = pd.get_dummies(df['Quad'])
features = pd.concat([features, quads], axis=1)
target = data['Angle']

# Feature normalisation
scaler = StandardScaler()
features = scaler.fit_transform(features)
target = scaler.fit_transform(np.array(target).reshape(-1, 1))

# Split
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state = 10)

# Linear Regression ANN
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=10, kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu', input_dim=15))
model.add(tf.keras.layers.Dense(units=1, kernel_initializer='normal', activation='linear'))


def soft_acc(y_true, y_pred):
    '''
    accuracy metric callback
    '''
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


# Compile, train, predict
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mse', soft_acc])
history = model.fit(features_train, target_train, epochs=20, validation_split=0.2, batch_size=100)
predict = model.predict(features_test).flatten()

# Plots
history_df = pd.DataFrame(history.history)
plt.plot(history_df.loc[:, ['loss']], color='blue', label='Training loss')
plt.plot(history_df.loc[:, ['val_loss']], color='green', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="best")
plt.show()

plt.figure()
plt.plot(history_df.loc[:, ['soft_acc']], color='blue', label='Training accuracy')
plt.plot(history_df.loc[:, ['val_soft_acc']], color='green', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.scatter(target_test[::20], predict[::20], c='k', s=0.5, marker='x')
ax.set(xlabel='True Value [arb]', ylabel='Predicted Value [arb]', title='Linear Regression Prediction')
plt.show()
print(target_test, predict)
