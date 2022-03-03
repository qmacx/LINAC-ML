import pandas as pd
import numpy as np
import os, glob

# Data cleaning
qmx = glob.glob('./data/QM*x/Cx.txt')
qmy = glob.glob('./data/QM*y/Cy.txt')
bx = glob.glob('./data/B*x/Cx.txt')
by = glob.glob('./data/B*y/Cy.txt')

qmx_paths = [x for x in qmx]
qmy_paths = [x for x in qmy]
bx_paths = [x for x in bx]
by_paths = [x for x in by]

OTRindex = np.arange(0, 23)
OTRcols = ['OTR{}'.format(x) for x in OTRindex]

df = pd.DataFrame()
for i in range(len(qmx_paths)):
    x = pd.read_csv(qmx_paths[i])
    x.columns=[OTRcols]
    x = x.rename(columns={'OTR0': 'dx'}) 
    x['element'] = np.ones_like(len(x['dx']))*i 
    
    y = pd.read_csv(qmy_paths[i])
    y.columns=[OTRcols]
    y = y.rename(columns={'OTR0': 'dy'}) 
    y['element'] = np.ones_like(len(y['dy']))*i 
    
    new_df = pd.concat([x, y], axis=0)
    new_df = new_df.fillna(0)
    df = pd.concat([df, new_df], axis=0)

for i in range(len(bx_paths)):
    x = pd.read_csv(bx_paths[i])
    x.columns=[OTRcols]
    x = x.rename(columns={'OTR0': 'dx'}) 
    x['element'] = np.ones_like(len(x['dx']))*i 
    
    y = pd.read_csv(by_paths[i])
    y.columns=[OTRcols]
    y = y.rename(columns={'OTR0': 'dy'}) 
    y['element'] = np.ones_like(len(y['dy']))*i 
    
    new_df = pd.concat([x, y], axis=0)
    new_df = new_df.fillna(0)
    df = pd.concat([df, new_df], axis=0)

print(df.dtypes)
print(df['OTR22'])
# Removing unwanted string features in columns

#for i in range(len(df['OTR22'])):













df.to_csv('./data/data.csv')
