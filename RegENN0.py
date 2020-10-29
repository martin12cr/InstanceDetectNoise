
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from scipy.stats import rankdata


### INPUT
# data=data without response variable
# response= response variable
# alpha= parameter to find noise
# k= number of neihgboors

### OUTPUT
# vector indicating which instances has noisy  in the response variable

def RegENN0(data, response, k=9, alfa=5):
    
    # find position of response variable
    #number=train_data.columns.get_loc(pos)
    
    # Delete the response from train_data
    #data=train_data.loc[:, train_data.columns != number]
    #response=train_data.loc[:,number]
    
    # Function to get the euclidean distance
    def distan(x,data):
        res=(x-data)**2
        return(np.sum(res,1)**0.5)

    # Function to get position from smallest to bigger at row
    def idxSize(x): 
        idx = rankdata(x, method='ordinal')
        return(idx)

    # Function to get indexes of smallest values. 
    def idxSmall(x, k): 
        idx = np.argpartition(x,k+1)
        return (idx)

    # Function to get mean and standart deviation.
    def stat(x, response): 
        m = np.mean(response[x])
        s = np.std(response[x])
        return (m,s)
    
    # USAR OTRO ALGORITMO PARA PREDICCION DE MEDIA 
   
    # Apply function to get distances of every instance
    distances=np.apply_along_axis(distan, 1, data, data)

    # Identify the size position of each value at row
    idxSizeDat=np.apply_along_axis(idxSize, 1, distances)

    # Change the value of the diagonal for the highest number at row in size position data
    np.fill_diagonal(idxSizeDat, idxSizeDat.shape[0]+50)

    # Get the index of the smallest values at the row in size position data
    idxF=np.apply_along_axis(idxSmall, 1, idxSizeDat, k)[:,0:k]

    # Get mean and standart deviation
    statis=np.apply_along_axis(stat, 1, idxF, response)


    # Get noise
    noise=np.absolute(statis[:,0]-response)> (alfa*statis[:,1])  
    noisef=[1.0 if  i == True else 0 for i in noise]
    noisef=np.array(noisef)
    
    return(noisef)

