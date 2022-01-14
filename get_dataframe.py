from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os, glob
import h5py

# script to create dataframes from all available h5 files in directory

base = 'XFELTransportLineRun_hdf5/XFELTransportLineRun_slan.h5'
paths = glob.glob("./data/*_"+base)
betapaths = glob.glob('./data/betavals-*.csv')
twisspaths = glob.glob('./data/*-data-df.csv')


if __name__ == '__main__':
    completedf = pd.DataFrame() # initiates empty dataframe which will hold labels and feature
    for i in range(len(betapaths)):
        betacx, betacy, alphacx, alphacy = [], [], [], [] 
        divergence, Cx, Cy, emittence = [], [], [], []
        dataset = pd.read_csv(betapaths[i]) 
        for j, path in enumerate(twisspaths):
            data = pd.read_csv('./data/dataframe{}.csv'.format(j)) # data from csv created in generate_data
            diverge = data['Sxp'] + data['Syp'] # creates a divergence array for the current input beta values
            Cx.append(np.array(data['Cx'])[2414])
            Cy.append(np.array(data['Cy'])[2414])
            betacx.append(np.array(data['betacx'])[2414]) # stores the exit beta values in a list
            betacy.append(np.array(data['betacy'])[2414])
            alphacx.append(np.array(data['alphacx'])[2414]) # stores the exit beta values in a list
            alphacy.append(np.array(data['alphacy'])[2414])
            emittence.append(np.array(data['total_ecn'])[2414])
            divergence.append(np.array(diverge)[18]) 
        
        dataset['Centroid_x'] = np.array(Cx)
        dataset['Centroid_y'] = np.array(Cy)
        dataset['Radial Centroid'] = np.sqrt((np.array(Cx)**2 + np.array(Cy)**2))
        dataset['betax_exit'] = np.array(betacx) # column containing exit beta values for each input pair
        dataset['betay_exit'] = np.array(betacy)
        dataset['alphax_exit'] = np.array(alphacx)
        dataset['alphay_exit'] = np.array(alphacy)
        dataset['divergence'] = np.array(divergence) 
        dataset['ecn_summed'] = np.array(emittence)       

    completedf = completedf.append(dataset).drop('Unnamed: 0', axis=1).fillna(0)
    completedf.to_csv('completed_data.csv') # dataframe containing sets of input and output values
