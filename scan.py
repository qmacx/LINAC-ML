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
beta_x = np.linspace(nbetax*0.99, nbetax*1.01, 2)
beta_y = np.linspace(nbetay*0.99, nbetay*1.01, 2)
angleb1 = np.linspace(nab1*0.995, nab1*1.005, 2) # varied by 0.5% each since b1=b4 and b2=b3 => 1% overall variation
angleb2 = np.linspace(nab2*0.995, nab2*1.005, 2)
angleb3 = np.linspace(nab3*0.995, nab3*1.005, 2)
angleb4 = np.linspace(nab4*0.995, nab4*1.005, 2)

if __name__ == '__main__':
    labels = grid_scan(beta_x, beta_y, angleb1, angleb2) # can be changed to beta_scan, chicane_scan
    #labels = pd.read_csv('./data/scanned_values.csv')
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


