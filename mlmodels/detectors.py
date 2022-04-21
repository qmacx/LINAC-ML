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

# File used for checking which OTR screens were relevant and should be kept

data = pd.read_csv('../data/data.csv')
OTRindex = np.arange(0, len(data.columns)-3)
cols = ['Quad', 'Labels', 'Angle']
OTRcols = ['OTR{}'.format(x) for x in OTRindex]
cols.extend(OTRcols)
data.columns=cols
data['OTR42'][13986] = -0.00022418886038814653

# removing chicane data
chicaneids = np.arange(13, 32) 
chicanecols = ['OTR{}'.format(x) for x in chicaneids]
chicanerows = ['B1', 'B2', 'B3', 'B4']
data = data.drop(chicanecols, axis=1)
data = data[~data.Quad.isin(chicanerows)]
print(data.head())
print(data.columns)


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
clf_target = pd.get_dummies(df['Quad'])
features = df.drop(['Labels', 'Quad', 'Angle'], axis=1)


def model_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


def build_model(features, target):
    
    inputdims = len(features.columns)
    outputdims = len(target.columns)
    
    # Split
    features = scaler.fit_transform(features)
    X_train, X_test, Yclf_train, Yclf_test = train_test_split(features, clf_target, test_size=0.20, random_state = 10)
    
    # DNN model building
    inputs = tf.keras.layers.Input(shape=(inputdims,))
    hidden1 = tf.keras.layers.Dense(units=inputdims * 2/3, kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')(inputs)
    hidden2 = tf.keras.layers.Dense(units=inputdims * 2/3, kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')(hidden1)
    hidden3 = tf.keras.layers.Dense(units=inputdims * 2/3, kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')(hidden2)
    #dropout = tf.keras.layers.Dropout(0.05)(hidden2, training=True)
    clf_outputs = tf.keras.layers.Dense(units=outputdims, kernel_initializer=tf.keras.initializers.glorot_normal(), activation='softmax')(hidden3)
    dnn = tf.keras.models.Model(inputs=inputs, outputs=clf_outputs)

    # compile, train, predict
    stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.001, patience=20, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    dnn.compile(loss=['categorical_crossentropy', 'mse'], metrics=[model_acc], optimizer=opt)
    history = dnn.fit(X_train, Yclf_train, batch_size=100, epochs=1000, validation_split=0.33)
    clf_pred = dnn.predict(X_test)
    clf_pred = np.argmax(clf_pred, axis=1)

    # results
    mapping = {'QM1_dx': 0, 'QM1_dy': 1, 'QM2_dx': 2, 'QM2_dy': 3, 'QM3_dx': 4, 'QM3_dy': 5,
    'QM4_dx': 6, 'QM4_dy': 7, 'QM5_dx': 8, 'QM5_dy': 9, 'QM6_dx': 10, 'QM6_dy': 11}

    keys = mapping.keys()
    xlabels = list(keys)
    Yclf_test = Yclf_test.idxmax(axis=1)
    Yclf_test = [mapping[i] for i in Yclf_test] # replaces strings with mapped value
    clf_acc = accuracy_score(Yclf_test, clf_pred)

    print('Classifier Accuracy: {:.4f}%'.format(clf_acc*100))

    return [clf_acc, inputdims]

# Determining best number of detectors

ids = np.arange(2, 42)
quadcols = ['OTR{}'.format(x) for x in ids if x not in chicaneids][::-1]
print(quadcols)
print(len(np.array(quadcols)))

results = []
for i in quadcols:
    features = features.drop([i], axis=1)
    print(features.head())
    results.append(build_model(features, clf_target))

print(results) # can drop [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 40, 41], this is determined by visually inspecting the array containing accuracy and number of features, each detector is represented as 2 separate OTRs (e.g the first detector is OTR0(dx) and OTR1(dy)), so if the difference in 1 pair offered no major improvement over the following pair, the first pair was removed leaving the pair (or OTR) which lies deepest in the beamline. 
