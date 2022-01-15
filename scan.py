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


beta_x = np.linspace(1, 2, 2)
beta_y = np.linspace(1, 2, 2)
pitchb1 = np.linspace(1, 2, 2)
pitchb2 = np.linspace(1, 2, 2)
pitchb3 = np.linspace(1, 2, 2)
pitchb4 = np.linspace(1, 2, 2)

paths = glob.glob('./data/dataframe*.csv')


# this counter is to avoid overwriting current files in directory, simply a workaround to the naming system I used
if len(paths) > 0:
    counter = len(paths)
else:
    counter = 0 


def grid_scan(bx, by, pb1, pb2, pb3, pb4, c):
   
    """ 
    Scans beta twiss values + chosen section
    """

    counter = c 
    for x in bx:
        stringx = "s/beta_x = .*/beta_x = %s/"%(x)
        os.system("sed -i '%s' XFELTransportLineRun.ele"%(stringx))
        for y in by:
            stringy = "s/beta_y = .*/beta_y = %s/"%(y)
            os.system("sed -i '%s' XFELTransportLineRun.ele"%(stringy))

            chicane = chicane_scan(x, y, pitchb1, pitchb2, pitchb3, pitchb4, counter)

    chicane.to_csv('./data/betavals-{}.csv'.format(date)) 


grid_scan(beta_x, beta_y, pitchb1, pitchb2, pitchb3, pitchb4, counter)
