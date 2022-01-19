from scan_functions import *
import numpy as np
import os, sys, glob, h5py
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import datetime

# nominal values
nbetax = 5.7598
nbetay = 4.5713
nab1 = 0.001
nab2 = -0.001
nab3 = -0.001
nab4 = 0.001

# must be equal length arrays
<<<<<<< HEAD
beta_x = np.linspace(nbetax*0.99, nbetax*1.01, 4)
beta_y = np.linspace(nbetay*0.99, nbetay*1.01, 4)
angleb1 = np.linspace(nab1*0.95, nab1*1.05, 100)
angleb2 = np.linspace(nab2*0.95, nab2*1.05, 100)

# script to create dataframes from all available h5 files in directory


if __name__ == '__main__':
    labels = chicane_scan(nbetax, nbetay, angleb1, angleb2) 
    paths = glob.glob("./data/dataframe*.csv")
    divergence = []
    emittence = []
    centroid = []
    for i in range(len(paths)):
        dataset = pd.read_csv(paths[i])
        diverge = dataset['Sxp'] + dataset['Syp']
        divergence.append(np.array(diverge)[22]) 
        
        ex0 = np.array(dataset['enx'])[0]
        ey0 = np.array(dataset['eny'])[0]
        ex = np.array(dataset['enx'])[2418] 
        ey = np.array(dataset['eny'])[2418]
        emittence.append(np.sqrt((ex-ex0)**2 + (ey-ey0)**2))
        
        cx = np.array(dataset['Cx'])[2418]
        cy = np.array(dataset['Cy'])[2418]
        centroid.append(np.sqrt(cx**2 + cy**2))
        
   
    features = pd.DataFrame(np.column_stack([divergence, emittence, centroid]), columns=['divergence', 'emittence', 'centroid'])
    training_data = pd.concat([labels.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    training_data.to_csv('./data/completed_data.csv') # dataframe containing sets of input and output values


