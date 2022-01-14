import numpy as np
import os, sys, glob, h5py
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

now = datetime.now()
date = str(now.strftime("%d-%m-%Y-%H:%M"))

nominalx = 5.7598
nominaly = 4.5713

coupled_vals = {'betax_in': [], 'betay_in': []}

beta_x = np.linspace(nominalx*0.1, nominalx*1.2, 2)
beta_y = np.linspace(nominaly*0.1 ,nominalx*1.2, 2)
x, y = np.meshgrid(beta_x, beta_y)

base = './data/XFELTransportLineRun_hdf5/XFELTransportLineRun_slan.h5'
paths = glob.glob('./data/dataframe*.csv')


def clean_data(path, n):
    """ 
    Returns dataframes containing only the relevant parameters using available paths in project directory
    """
    file = h5py.File(path)['page1']['columns']
    keys = file.keys()
    data, dataavg, clean_keys, clean_keysavg = [], [], [], []

    for key in keys:
        if "Slice" not in key:
            if "Ave" not in key:
                data.append(file[key])
                clean_keys.append(key)
            elif "Ave" in key:
                dataavg.append(file[key])
                clean_keysavg.append(key)

    rawdf = pd.DataFrame(np.array(data), index=clean_keys).transpose()
    rawdfavg = pd.DataFrame(np.array(dataavg), index=clean_keysavg).transpose()
    rawdf = rawdf.drop(['ElementName', 'particles'], axis=1)
    rawdf['total_ecn'] = rawdf['ecnx'] + rawdf['ecny']
    
    xp = np.array(pd.read_csv("./twiss_ascii/XP_XFELTransportLineRun.txt")).reshape(1000,)
    x = np.array(pd.read_csv("./twiss_ascii/X_XFELTransportLineRun.txt")).reshape(1000,)
    yp = np.array(pd.read_csv("./twiss_ascii/YP_XFELTransportLineRun.txt")).reshape(1000,)
    y = np.array(pd.read_csv("./twiss_ascii/Y_XFELTransportLineRun.txt")).reshape(1000,)
    twiss = pd.read_csv("./twiss_ascii/twiss_parameter_slan_XFELTransportLineRun.txt", sep='\t', header=0)
    diverge = pd.read_csv("./twiss_ascii/diverg_sig_XFELTransportLineRun.txt", sep='\t', header=0)
    
    phase = pd.DataFrame(np.array([x, xp, y, yp]), index=['x', 'xp', 'y', 'yp']).T
    frames = [rawdf, rawdfavg, phase, twiss, diverge] 
    df = pd.concat(frames, axis=1)
    df.to_csv('./data/dataframe{}.csv'.format(n))

    return df




def makedir(path):
  try:
    os.makedirs(path)
  except OSError:
    pass



# checks if there is already data in the directory and avoids overwriting
if len(paths) > 0:
    counter = len(paths) 
else:
    counter = 0



if __name__ == '__main__':
    for i in range(len(x)):
        stringx = "s/beta_x = .*/beta_x = %s/"%(beta_x[i])
        for j in range(len(y)):
            makedir('./data')
            stringy = "s/beta_y = .*/beta_y = %s/"%(beta_y[j])
            
            os.system("sed -i '%s' XFELTransportLineRun.ele"%(stringx))
            os.system("sed -i '%s' XFELTransportLineRun.ele"%(stringy))
            os.system("elegant XFELTransportLineRun.ele")
            os.system("python elegant2hdf5.py")
            
            coupled_vals['betax_in'].append(beta_x[i]) # stores pairs of beta values
            coupled_vals['betay_in'].append(beta_y[j])
            
            clean_data(base, counter) # creates output files for twiss params
            counter += 1

    pd.DataFrame.from_dict(coupled_vals).to_csv('./data/betavals-{}.csv'.format(date)) # creates a csv containing parameter pairs per df which are later matched up with singular output values

