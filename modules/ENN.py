
# coding: utf-8

# In[ ]:


##################  ENN  CLASSIFICATION ######################################

### INPUT
# data=data without response variable
# response= response variable
# alpha= parameter to find noise
# k= number of neihgboors

### OUTPUT
# vector indicating which instances has noisy  in the response variable

#data=data1
#response=response1
#k=9
#alfa=5

import pandas as pd
import numpy as np
from scipy.stats import rankdata
import pandas as pd
import numpy as np
from scipy.stats import rankdata

def ENN(data, response, k=9):
    
    # find position of response variable
    #number=train_data.columns.get_loc(pos)
    
    # Delete the response from train_data
    #data=train_data.loc[:, train_data.columns != number]
    #response=train_data.loc[:,number]
    
    # Creating replicated files 
    response=np.array(response)
    dataMovil=data
    responseMovil=response
    cont=0
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

    # Function to get prediction majority vote
    def predic(x, response): 
        neigh=response[x]
        frec=np.unique(neigh, return_counts=True)    
        pre=frec[0][np.argmax(frec[1])]
        return(pre)

    # Vector to save if the instance is noisy
    noisef=np.empty([0])
    indexDel=np.empty([0], dtype=int)
    size=response.shape[0]
    # For each instance
    for i in range(size):
    
        # distances
        distances=distan(data[i,:],dataMovil)   

        # Identify the size position of each value at row
        idxSizeDat=idxSize(distances)
       
        # Cambiar i por un valor alto como idxSizeDat.shape[0]+50
        idxSizeDat[cont]=100

        #  Get the index of the smallest values at the row in size position data
        idxF=idxSmall(idxSizeDat, k)[0:k]

        # Get mean and standart deviation
        pre=predic(idxF, responseMovil)
        
        # Count of index for data with deleted instances
        cont=cont+1 
        
        # Get noise
        #noise=response[i]!=pre 
        if (int(response[i])!=int(pre)):
            # Detect noise
            noisef=np.append(noisef,1)
            
            # Detect index noise
            indexDel=np.append(indexDel,i)
            
            # Delete instances noisy
            dataMovil = np.delete(data, indexDel, axis=0)
            responseMovil=np.delete(response,indexDel)
            
            # rest to cont
            cont=cont-1
            
        else:
            noisef=np.append(noisef,0)
           
    return (noisef)

