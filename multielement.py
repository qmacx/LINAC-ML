import pandas as pd
from tensorflow import keras
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
from keras import layers

# DNN for classification of coupled quadrupole misalignments in the capture stage of the beamline

data = pd.read_csv('DXfirst3Quads.csv')
data.columns=['Quad', 'Q1', 'Q2', 'Q3', 'CxOTR1', 'CxOTR2', 'CxOTR3', 'CxOTR4', 'CxOTR5', 'CxOTR6']
df = data.copy()

# Machine Learning
features = df.drop(['Q1', 'Q2', 'Q3', 'Quad'], axis=1)
target = pd.get_dummies(df['Quad'])
print(features)
print(target)

scaler = StandardScaler()
features = scaler.fit_transform(features)

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state = 10)

# model building
initializer = tf.keras.initializers.HeNormal()

dnn = tf.keras.models.Sequential()
dnn.add(tf.keras.layers.Dense(units=10, kernel_initializer=initializer, activation='relu', input_dim=6))
dnn.add(tf.keras.layers.Dense(units=10, kernel_initializer=initializer, activation='relu'))
dnn.add(tf.keras.layers.Dense(units=7, kernel_initializer=tf.keras.initializers.glorot_normal(), activation='softmax'))

# compiling
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
dnn.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.001, patience=20, verbose=0,
    mode='auto', baseline=None, restore_best_weights=False)


history = dnn.fit(features_train, target_train, batch_size=32, epochs=1000, validation_split=0.2, callbacks=[stop])


# plots
history_df = pd.DataFrame(history.history)
plt.plot(history_df.loc[:, ['loss']], color='blue', label='Training loss')
plt.plot(history_df.loc[:, ['val_loss']], color='green', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="best")
plt.show()

plt.figure()
plt.plot(history_df.loc[:, ['accuracy']], color='blue', label='Training accuracy')
plt.plot(history_df.loc[:, ['val_accuracy']], color='green', label='Validation accuracy')

plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

mapping = {'QM1': 0, 'QM12': 1, 'QM123': 2, 'QM13': 3, 'QM2': 4, 'QM23': 5, 'QM3': 6}
target_test = target_test.idxmax(axis=1)
target_test = [mapping[i] for i in target_test] # replaces strings with mapped value
predictions = np.argmax(dnn.predict(features_test), axis=1)
print('prediction accuracy: ', accuracy_score(target_test, predictions))

# confusion matrix
xlabels = ['QM1', 'QM12', 'QM123', 'QM13', 'QM2', 'QM23', 'QM3']
cm = confusion_matrix(target_test, predictions)
ax = sns.heatmap(cm, xticklabels=xlabels, yticklabels=xlabels, annot=True)
ax.set(title='Confusion Matrix for Multiclass DNN prediction', xlabel='Predicted Value', ylabel='True Value')
plt.show()
