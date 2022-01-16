import numpy as np
import os, sys, glob, h5py
import pandas as pd
from datetime import datetime


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
    
    xp = np.array(pd.read_csv("./twiss_ascii/XP_XFELTransportLineRun.txt")).reshape(10,)
    x = np.array(pd.read_csv("./twiss_ascii/X_XFELTransportLineRun.txt")).reshape(10,)
    yp = np.array(pd.read_csv("./twiss_ascii/YP_XFELTransportLineRun.txt")).reshape(10,)
    y = np.array(pd.read_csv("./twiss_ascii/Y_XFELTransportLineRun.txt")).reshape(10,)
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



def beta_scan(betax, betay):
    """
    Scans beta twiss values
    """

    base = './data/XFELTransportLineRun_hdf5/XFELTransportLineRun_slan.h5'
    coupled_vals = {'betax_in': [], 'betay_in': []}
    
    for x in betax:
        stringx = "s/beta_x = .*/beta_x = %s/"%(betax)
        os.system("sed -i '%s' XFELTransportLineRun.ele"%(stringx))
        for y in betay:
            stringy = "s/beta_y = .*/beta_y = %s/"%(betay)
            os.system("sed -i '%s' XFELTransportLineRun.ele"%(stringy))
            
            os.system("elegant XFELTransportLineRun.ele")
            os.system("python elegant2hdf5.py")
            os.system("./plot_twissV9.sh XFELTransportLineRun.slan XFELTransportLineRun.magn")

            coupled_vals['betax_in'].append(x) # stores pairs of beta values
            coupled_vals['betay_in'].append(y)
           
            clean_data(base, counter)  # creates output dataframe
            counter += 1

    pd.DataFrame.from_dict(coupled_vals).to_csv('./data/betavals-{}.csv'.format(date)) 



def chicane_scan(betax, betay, p1, p2, p3, p4, counter): 
    """
    Scans the pitch values for the chicane
    """

    base = './data/XFELTransportLineRun_hdf5/XFELTransportLineRun_slan.h5'
    chicane_vals = {'betax_in': [], 'betay_in': [], 'pitchb1': [], 'pitchb2': [], 'pitchb3': [], 'pitchb4': []}

    c = counter
    for pitchb1 in p1:
        stringb1 = "s/B1: = .*/B1: PITCH=%s,/"%(pitchb1)
        os.system("sed -i '%s' XFELTransportLineFinal.lte"%(stringb1))
        for pitchb2 in p1:
            stringb2 = "s/B2: = .*/B2: PITCH=%s,/"%(pitchb2)
            os.system("sed -i '%s' XFELTransportLineFinal.lte"%(stringb2))
            for pitchb3 in p3:
                stringb3 = "s/B3: = .*/B3: PITCH=%s,/"%(pitchb3)
                os.system("sed -i '%s' XFELTransportLineFinal.lte"%(stringb3))
                for pitchb4 in p4:
                    stringb4 = "s/B4: = .*/B4: PITCH=%s,/"%(pitchb4)
                    os.system("sed -i '%s' XFELTransportLineFinal.lte"%(stringb4))
                    
                    os.system("elegant XFELTransportLineRun.ele")
                    os.system("python elegant2hdf5.py")
                    os.system("./plot_twissV9.sh XFELTransportLineRun.slan XFELTransportLineRun.magn")
                   
                    chicane_vals['betax_in'].append(betax) 
                    chicane_vals['betay_in'].append(betay) 
                    chicane_vals['pitchb1'].append(pitchb1) 
                    chicane_vals['pitchb2'].append(pitchb2)
                    chicane_vals['pitchb3'].append(pitchb3)
                    chicane_vals['pitchb4'].append(pitchb4)
                    
                    clean_data(base, c) # creates output files for twiss params
                    c += 1

    chicane = pd.DataFrame.from_dict(chicane_vals)
    
    return chicane




def grid_scan(bx, by, pb1, pb2, pb3, pb4):
   
    """ 
    Scans beta twiss values (first 2 layers of nested loop) + chosen section
    """

    scannedvals = pd.DataFrame() # df to store all combinations of parameter scan
    outputs = glob.glob('./data/dataframe*.csv') # all dataframe paths containing output values
    labels = pd.DataFrame(columns=['betax', 'betay', 'pitch1', 'pitch2', 'pitch3', 'pitch4'])

    # added this check so that running the script consecutively doesn't overwrite current paths
    if len(outputs) > 0:
        out_count = len(outputs)
    else:
        out_count = 0 
         
    for x in bx: 
        stringx = "s/beta_x = .*/beta_x = %s/"%(x)
        os.system("sed -i '%s' XFELTransportLineRun.ele"%(stringx))
        for y in by: 
            stringy = "s/beta_y = .*/beta_y = %s/"%(y)
            os.system("sed -i '%s' XFELTransportLineRun.ele"%(stringy))

            scannedvals = scannedvals.append(chicane_scan(x, y, pb1, pb2, pb3, pb4, out_count))
            out_count += len(bx)**4 # necessary to not overwrite chicane outputs every time chicane_scan called 

    scannedvals.to_csv('./data/scanned_values.csv', index=False)

    return scannedvals
