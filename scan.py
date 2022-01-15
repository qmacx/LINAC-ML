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

            scannedvals = scannedvals.append(chicane_scan(x, y, pitchb1, pitchb2, pitchb3, pitchb4, out_count))
            out_count += len(bx)**4 # necessary to not overwrite chicane outputs every time chicane_scan called 

    scannedvals.to_csv('./data/scanned_values.csv', index=False)
    

grid_scan(beta_x, beta_y, pitchb1, pitchb2, pitchb3, pitchb4)
