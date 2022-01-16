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
    training_data = grid_scan(beta_x, beta_y, angleb1, angleb2) 
    paths = glob.glob("./data/dataframe*.csv")
    divergence = []
    for i in range(len(paths)):
        dataset = pd.read_csv(paths[i])
        diverge = dataset['Sxp'] + dataset['Syp']
        divergence.append(np.array(diverge)[22]) 
    
    training_data['divergence'] = divergence
    training_data.to_csv('./data/completed_data.csv') # dataframe containing sets of input and output values


