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


# Take the evolution of centroid and assign a label to that 
# labels should be quadrupole element number

data = pd.read_csv('DxDyfirst3Quads10000.csv')
data = data.drop(labels=range(30000, 39997), axis=0)
data.columns=['Label','Quad','Angle','CxOTR1','CyOTR1','CxOTR2','CyOTR2','CxOTR3','CyOTR3','CxOTR4','CyOTR4', 'CxOTR5','CyOTR5','CxOTR6','CyOTR6','CxOTR7','CyOTR7']

dfx = data[data['Label'] == 'Dx'] 
print(dfx[dfx['Quad'] == 'QM3'])
dfy = data[data['Label'] == 'Dy']
nm = data[data['Label'] == 'No_Misalign']

mapx = {'QM1': 'QM1_dx', 'QM2': 'QM2_dx', 'QM3': 'QM3_dx'}
mapy = {'QM1': 'QM1_dy', 'QM2': 'QM2_dy', 'QM3': 'QM3_dy'}

dfx['Quad'] = [mapx[i] for i in dfx['Quad']]
dfy['Quad'] = [mapy[i] for i in dfy['Quad']]

df = pd.concat([dfx, dfy, nm], axis=0)

# Machine Learning
features = df.drop(['Label', 'Quad', 'Angle'], axis=1)
#features = df.drop(['Quad', 'Angle'], axis=1)
target = pd.get_dummies(df['Quad'])

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state = 10)

# model building
initializer = tf.keras.initializers.HeNormal()

dnn = tf.keras.models.Sequential()
dnn.add(tf.keras.layers.Dense(units=10, kernel_initializer=initializer, activation='relu', input_dim=14))
dnn.add(tf.keras.layers.Dense(units=10, kernel_initializer=initializer, activation='relu'))
dnn.add(tf.keras.layers.Dense(units=7, kernel_initializer=tf.keras.initializers.glorot_normal(), activation='softmax')) # softmax clfn

# compiling
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
dnn.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.0001, patience=30, verbose=0,
    mode='auto', baseline=None, restore_best_weights=False)



def adaptive_learning(epoch):
    return 1 / (10000 * epoch + 1)


lr_scheduler = tf.keras.callbacks.LearningRateScheduler(adaptive_learning)


# training
history = dnn.fit(features_train, target_train, batch_size=32, epochs=5000, validation_split=0.2, callbacks=[stop])

# plots
plot_model(dnn, to_file='dnn_plot.png', show_shapes=True, show_layer_names=True)

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

mapping = {'No_Misalign': 0, 'QM1_dx': 1, 'QM1_dy': 2, 'QM2_dx': 3, 'QM2_dy': 4, 'QM3_dx': 5, 'QM3_dy': 6}
target_test = target_test.idxmax(axis=1) 
target_test = [mapping[i] for i in target_test] # replaces strings with mapped value
predictions = np.argmax(dnn.predict(features_test), axis=1)
print('prediction accuracy: ', accuracy_score(target_test, predictions))

# confusion matrix
xlabels = ['None', 'QM1x', 'QM1y', 'QM2x', 'QM2y', 'QM3x', 'QM3y'] 

cm = confusion_matrix(target_test, predictions)
ax = sns.heatmap(cm, xticklabels=xlabels, yticklabels=xlabels, annot=True)
ax.set(title='Confusion Matrix for Multiclass DNN prediction', xlabel='Predicted Value', ylabel='True Value')
plt.show()
