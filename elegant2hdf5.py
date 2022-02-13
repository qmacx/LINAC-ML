from IPython import get_ipython
import glob, os,sys, re
from datetime import datetime,timedelta
now = datetime.now()

# converts Elegant output files to hdf5 for Python analysis

TodayDate=str(now.strftime("%d-%m-%Y-%H:%M:%S:%f"))

#dir is not keyword
def makemydir(whatever):
  try:
    os.makedirs(whatever)
  except OSError:
    pass

FileBaseName='XFELTransportLineRun'
SavePath='./data/'+FileBaseName+'_hdf5/'

def FileNameSplit(DataFilesIn):
    DataFilesOut  = DataFilesIn.replace(".","_")
    
    return(DataFilesOut)


def ConvertSDDS2HDF5(FileBaseName,SavePath):
    DataFiles = glob.glob(FileBaseName+".*")

    for FileLoad in DataFiles:
        
        makemydir(SavePath)
        os.system('sdds2hdf %s %s.h5'%(FileLoad,SavePath+FileNameSplit(FileLoad)))

    return

ConvertSDDS2HDF5(FileBaseName,SavePath)
