from scan_functions import *
import numpy as np
import os, sys, glob, h5py
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import datetime

now = datetime.now()
date = str(now.strftime("%d-%m-%Y-%H:%M"))

nbetax = 5.7598
nbetay = 4.5713
nab1 = 0.001
nab2 = -0.001
nab3 = -0.001
nab4 = 0.001



# must be equal length arrays
beta_x = np.linspace(nbetax*0.99, nbetax*1.01, 2)
beta_y = np.linspace(nbetay*0.99, nbetay*1.01, 2)
angleb1 = np.linspace(nab1*0.9999, nab1*1.0001, 2)
angleb2 = np.linspace(nab2*0.9999, nab2*1.0001, 2)
angleb3 = np.linspace(nab3*0.9999, nab3*1.0001, 2)
angleb4 = np.linspace(nab4*0.9999, nab4*1.0001, 2)

# script to create dataframes from all available h5 files in directory


if __name__ == '__main__':
    labels = grid_scan(beta_x, beta_y, angleb1, angleb2) 
    paths = glob.glob("./data/dataframe*.csv")
    divergence = []
    emittence = []
    centroid = []
    for i in range(len(paths)):
        dataset = pd.read_csv(paths[i])
        diverge = dataset['Sxp'] + dataset['Syp']
        ex = dataset['enx'] - 4.0e-8
        ey = dataset['eny'] - 4.0e-8
        emit = np.sqrt(ex**2 + ey**2)

        cx = dataset['Cx']
        cy = dataset['Cy']
        cent = np.sqrt(cx**2 + cy**2)
        
        divergence.append(np.array(diverge)[22]) 
        emittence.append(np.array(emit)[2418])
        centroid.append(np.array(cent)[2418])
   
    features = pd.DataFrame(np.column_stack([divergence, emittence, centroid]), columns=['divergence', 'emittence', 'centroid'])
    training_data = pd.concat([labels, features], axis=1)
    training_data.to_csv('./data/completed_data.csv') # dataframe containing sets of input and output values


