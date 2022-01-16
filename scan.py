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
npb1 = 0.01
npb2 = 0.01
npb3 = 0.01
npb4 = 0.01



# must be equal length arrays
beta_x = np.linspace(nbetax*0.99, nbetax*1.01, 2)
beta_y = np.linspace(nbetay*0.99, nbetay*1.01, 2)
pitchb1 = np.linspace(npb1*0.99, npb1*1.01, 2)
pitchb2 = np.linspace(npb2*0.99, npb2*1.01, 2)
pitchb3 = np.linspace(npb3*0.99, npb3*1.01, 2)
pitchb4 = np.linspace(npb4*0.99, npb4*1.01, 2)

# script to create dataframes from all available h5 files in directory


if __name__ == '__main__':
    training_data = grid_scan(beta_x, beta_y, pitchb1, pitchb2, pitchb3, pitchb4) 
    paths = glob.glob("./data/dataframe*.csv")
    divergence = []
    for i in range(len(paths)):
        dataset = pd.read_csv(paths[i])
        diverge = dataset['Sxp'] + dataset['Syp']
        divergence.append(np.array(diverge)[22]) 
    
    training_data['divergence'] = divergence
    training_data.to_csv('./data/completed_data.csv') # dataframe containing sets of input and output values


