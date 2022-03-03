import pandas as pd
import numpy as np
import os, glob

# Data cleaning
qmx = glob.glob('./data/QM*x/Cx.txt')
qmy = glob.glob('./data/QM*y/Cy.txt')
pmqx = qmx[0:3]
pmqy = qmy[0:3]
emqx = qmx[3::]
emqy = qmy[3::]

bx = glob.glob('./data/B*x/Cx.txt')
by = glob.glob('./data/B*y/Cy.txt')
bx_paths = [x for x in bx]
by_paths = [x for x in by]

OTRindex = np.arange(0, 23)
OTRcols = ['OTR{}'.format(x) for x in OTRindex]

df = pd.DataFrame()
c = 0
for i in range(len(pmqx)):
    x = pd.read_csv(pmqx[i])
    x.columns=[OTRcols]
    x = x.rename(columns={'OTR0': 'dx'}) 
    x['element'] = np.ones_like(len(x['dx']))*c 
    
    y = pd.read_csv(pmqy[i])
    y.columns=[OTRcols]
    y = y.rename(columns={'OTR0': 'dy'}) 
    y['element'] = np.ones_like(len(y['dy']))*c

    c += 1
    new_df = pd.concat([x, y], axis=0)
    new_df = new_df.fillna(0)
    df = pd.concat([df, new_df], axis=0)


for i in range(len(bx_paths)):
    x = pd.read_csv(bx_paths[i])
    x.columns=[OTRcols]
    x = x.rename(columns={'OTR0': 'dx'}) 
    x['element'] = np.ones_like(len(x['dx']))*c
    
    y = pd.read_csv(by_paths[i])
    y.columns=[OTRcols]
    y = y.rename(columns={'OTR0': 'dy'}) 
    y['element'] = np.ones_like(len(y['dy']))*c
    
    c += 1
    new_df = pd.concat([x, y], axis=0)
    new_df = new_df.fillna(0)
    df = pd.concat([df, new_df], axis=0)


for i in range(len(emqx)):
    x = pd.read_csv(emqx[i])
    x.columns=[OTRcols]
    x = x.rename(columns={'OTR0': 'dx'}) 
    x['element'] = np.ones_like(len(x['dx']))*c
    
    y = pd.read_csv(emqy[i])
    y.columns=[OTRcols]
    y = y.rename(columns={'OTR0': 'dy'}) 
    y['element'] = np.ones_like(len(y['dy']))*c
    
    c += 1
    new_df = pd.concat([x, y], axis=0)
    new_df = new_df.fillna(0)
    df = pd.concat([df, new_df], axis=0)


df.to_csv('./data/data.csv')
