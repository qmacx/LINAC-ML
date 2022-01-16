from scan_functions import *
import numpy as np
import os, sys, glob, h5py
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import datetime

now = datetime.now()
date = str(now.strftime("%d-%m-%Y-%H:%M"))

nominalx = 5.7598
nominaly = 4.5713

# must be equal length arrays
beta_x = np.linspace(1, 2, 2)
beta_y = np.linspace(1, 2, 2)
pitchb1 = np.linspace(1, 2, 2)
pitchb2 = np.linspace(1, 2, 2)
pitchb3 = np.linspace(1, 2, 2)
pitchb4 = np.linspace(1, 2, 2)

# script to create dataframes from all available h5 files in directory

base = './data/XFELTransportLineRun_hdf5/XFELTransportLineRun_slan.h5'
makedir(base)
paths = glob.glob("./data/dataframe*.csv")

if __name__ == '__main__':
    labels = grid_scan(beta_x, beta_y, pitchb1, pitchb2, pitchb3, pitchb4) # scans the parameters
    divergence = []
    for i in range(len(paths)):
        dataset = pd.read_csv(paths[i]) 
        diverge = data['Sxp'] + data['Syp'] 
        divergence.append(np.array(diverge)[18]) 
    
    feature = pd.DataFrame(divergence, columns=['divergence']) # initiates empty dataframe which will hold feature (divergence)
    training_data = pd.concat([labels, feature])
    training_data.to_csv('completed_data.csv') # dataframe containing sets of input and output values

    

