#import tarfile
from zipfile import ZipFile 
from six.moves import urllib
from ipywidgets import FloatProgress
from IPython.display import display
import os
import pandas as pd
import numpy as np


DATA_PATH = os.path.join('Data','emnist')
DATA_URL = "https://storage.googleapis.com/ml_data_pib/char_recognition/emnist.zip"

def LoadData(filename,path=DATA_PATH):
    file_path =  os.path.join(path,filename)
    df = pd.read_csv(file_path,header=None)
    data = df.iloc[:,1:]
    target = df.iloc[:,0]
    del df
    data = _rotateAll(data)
    return target,data

# Data set has output label as integer. Load mapping of actual Character to its integer label in data
def LoadMapping(filename,path = DATA_PATH):
    map_dict = {}
    file_path =  os.path.join(path,filename)
    with open(file_path) as f:
        for line in f.readlines():
            line = line.split(' ')
            map_dict[int(line[0])] = str(chr(int(line[1])))
    return map_dict

def _rotateAll(data):
    def rotate(image):
        image = image.reshape([28, 28])
        image = image = image.transpose()
        return image.reshape([28 * 28])
    
    return np.apply_along_axis(rotate,1,data)

class Pbar:
    pbar = None
    @classmethod
    def getPbar(cls,total_size):
        if cls.pbar == None:
#            print('Get new Progress Bar')
            cls.pbar = FloatProgress(min=0, max=total_size)
            display(cls.pbar)
        return cls.pbar
            
def _show_progress(count, block_size, total_size):
    #print(count, block_size, total_size)
    pBar = Pbar.getPbar(total_size/block_size)
    pBar.value = count
    
def FetchData(data_url=DATA_URL, data_path=DATA_PATH):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    zip_path = os.path.join(data_path, "emnist.zip")
    if not os.path.isfile(zip_path):
        urllib.request.urlretrieve(data_url, zip_path, _show_progress)
    else:
        print("Data file already exist")
        
    with ZipFile(zip_path, 'r') as zip:
        print('Extracting all the files now...') 
        zip.extractall(data_path)
        print('Done!')
        zip.printdir()



