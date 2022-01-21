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

data = pd.read_csv('PITCHfirst3Quads10000.csv')
data.columns=['Quad','Angle','CxOTR1','CyOTR1','CxOTR2','CyOTR2','CxOTR3','CyOTR3','CxOTR4','CyOTR4', 'CxOTR5','CyOTR5','CxOTR6','CyOTR6','CxOTR7','CyOTR7']
df = data.copy()

features = df.drop(['Quad', 'Angle'], axis=1)
target = pd.get_dummies(df['Quad']) 

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state = 50)

# model building
dnn = tf.keras.models.Sequential()
dnn.add(tf.keras.layers.Dense(units=32, kernel_initializer=initializers.Ones(), activation='relu', input_dim=14))
dnn.add(tf.keras.layers.Dense(units=16, kernel_initializer='uniform', activation='relu'))
dnn.add(tf.keras.layers.Dense(units=8, kernel_initializer='uniform', activation='relu'))
dnn.add(tf.keras.layers.Dense(units=4, kernel_initializer='uniform', activation='softmax')) # softmax clfn

# compiling
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
dnn.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# training
history = dnn.fit(features_train, target_train, batch_size=32, epochs=150, validation_split=0.2)

# plots
plot_model(dnn, to_file='dnn_plot.png', show_shapes=True, show_layer_names=True)

history_df = pd.DataFrame(history.history)
plt.plot(history_df.loc[:, ['loss']], color='blue', label='Training loss')
plt.plot(history_df.loc[:, ['val_loss']], color='green', label='Validation loss')
plt.title('Training and Validation loss 32-16-8-3, softmax')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="best")
plt.show()

plt.figure()
plt.plot(history_df.loc[:, ['accuracy']], color='blue', label='Training accuracy')
plt.plot(history_df.loc[:, ['val_accuracy']], color='green', label='Validation accuracy')

plt.title('Training and Validation accuracy 32-16-8-3, softmax')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print('target', target_test, target_test.shape)
predictions = pd.get_dummies(np.argmax(dnn.predict(features_test), axis=1))
print('target', target_test, target_test.shape)
print('predictions', predictions, predictions.shape)
print(accuracy_score(target_test, predictions))
