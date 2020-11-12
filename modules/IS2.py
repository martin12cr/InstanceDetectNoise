
# coding: utf-8

# # RegENN01

# In[ ]:


################## RegENN NOT simultaneouos  ######################################

### INPUT
# data=data without response variable
# response= response variable
# alpha= parameter to find noise
# k= number of neihgboors

### OUTPUT
# vector indicating which instances has noisy  in the response variable

import pandas as pd
import numpy as np
from scipy.stats import rankdata
import pandas as pd
import numpy as np
from scipy.stats import rankdata

def RegENN01(data, response,alfa=5, k=9):
    
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

    # Function to get mean and standart deviation.
    def stat(x, response): 
        m = np.mean(response[x])
        s = np.std(response[x])
        return (m,s)

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
        statis=stat(idxF, responseMovil)
        
        # Count of index for data with deleted instances
        cont=cont+1 
        
        # Get noise
        noise=np.absolute(statis[0]-response[i])> (alfa*statis[1])  
        if (noise)==True:
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
    


# # RegENN03

# In[ ]:


################## RegENN03 ALFA* MAE VECINOS..... ######################################

### INPUT
# data=data without response variable
# response= response variable
# alpha= parameter to find noise
# k= number of neihgboors

### OUTPUT
# vector indicating which instances has noisy  in the response variable

import pandas as pd
import numpy as np
from scipy.stats import rankdata
import pandas as pd
import numpy as np
from scipy.stats import rankdata

def RegENN03(data, response, k=9, alfa=5):
    
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

    # Get Noise of the neighboors
    error=np.absolute(statis[:,0]-response)
    MAE=np.apply_along_axis(stat, 1, idxF, error)
    # pred=np.apply_along_axis(stat, 1, idxF, statis[:,0]) # PRUEBA3 CON DESV PRED
    
    # Get noise
    # noise=np.absolute(statis[:,0]-response)> ((alfa*MAE[:,1])+MAE[:,0]) # PRUEBA2 CON DESV
    # noise=np.absolute(statis[:,0]-response)> ((alfa*pred[:,1]) # PRUEBA3 CON DESV
    noise=np.absolute(statis[:,0]-response)> (alfa*MAE[:,0]) 
    noisef=[1.0 if  i == True else 0 for i in noise]
    noisef=np.array(noisef)
    
    # Get error
    # error=np.absolute(statis[:,0]-response)
    
    return noisef


# # RegBAG BAGGING

# In[ ]:


################## RegBAG  ######################################

### INPUT
# data=data without response variable
# response= response variable
# alpha= parameter to find noise
# k= number of neihgboors
# numBag=number of samples in bagging 
# samSize= number of elements per sample. 

### OUTPUT
# vector indicating which instances has noise  in the response variable

import pandas as pd
import numpy as np
from scipy.stats import rankdata
import pandas as pd
import numpy as np
from scipy.stats import rankdata

def RegBAG(data, response,alfa=5, k=9, numBag=10, samSize=1): 
    
    # find position of response variable
    #number=train_data.columns.get_loc(pos)
    
    # Delete the response from train_data
    #data=train_data.loc[:, train_data.columns != number]
    #response=train_data.loc[:,number]
    
    # Size sample bagging
    samSize=samSize*data.shape[0]
    
    # Creating files 
    response=np.array(response)                          # convert response to array 
    noiseMatrix=np.empty([data.shape[0]]).reshape(-1,1)  # creating matriz to save results
    cont=0                                               # count variable
    
    
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

    # row number of dataframe
    size=response.shape[0]
    
    # Bagging loop
    #for k in range(numBag):
    while cont<numBag:
        
        # Vector to save noisy instance and the index of noisy instances
        noisef=np.empty([0])                # vector
        indexDel=np.empty([0], dtype=int)   # index
        
        # bagging index selection
        consec=np.r_[0:data.shape[0]]
        r=np.random.choice(a=consec, size=samSize, replace=True)
        cont=cont+1

        # Apply for each instance
        for i in range(size):
    
            # Exclude from index the instance in process 
            r2= r[r != i]
            
            # Exclude the index of noisy  instances 
            r2=np.setdiff1d(r2,indexDel)
        
        
            # Select bagging values without instances noisy and the instances in process
            dataMovil = data[r2,:]
            responseMovil=response[r2]

            # distances
            distances=distan(data[i,:],dataMovil)   

            # Identify the size position of each value at row
            idxSizeDat=idxSize(distances)
       
            # Cambiar i por un valor alto como idxSizeDat.shape[0]+50
            #idxSizeDat[cont]=100

            #  Get the index of the smallest values at the row in size position data
            idxF=idxSmall(idxSizeDat, k)[0:k]

            # Get mean and standart deviation
            statis=stat(idxF, responseMovil)
        
            # Get noise
            noise=np.absolute(statis[0]-response[i])> (alfa*statis[1])  
            if (noise)==True:
                # Detect noise
                noisef=np.append(noisef,1)
            
                # Detect index noise
                indexDel=np.append(indexDel,i)
            
            else:
                noisef=np.append(noisef,0)
            
        # Save matrix    
        noiseMatrix=np.append(noiseMatrix, noisef.reshape(-1,1), axis=1)
    
    # Delete first column
    noiseMatrix=noiseMatrix[:,1:]   
    
    # Creating noise vector 
    noiseF2=np.where(np.sum(noiseMatrix, axis=1) > int(numBag/2), 1, 0)
    
    return (noiseF2)


# # ENN FOR CLASSIFICATION (este solo se ocupa para  DiscENN)

# In[4]:


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


# # DiscENN_SIN OPTIMIZADOR 

# In[3]:


##################### DiscENN_SIN OPTIMIZADOR ############################

from sklearn.metrics import f1_score
import sklearn
from sklearn.preprocessing import KBinsDiscretizer

def DiscENN(data, response, strat, bins,  k=9): 
    # Transform response to apply KBins
    responseTem=np.array(response).reshape(-1, 1)

    # Apply KBins
    kbins = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strat)
    response_trans = kbins.fit_transform(responseTem)

    # Apply ENN
    #ENN = EditedNearestNeighbours(n_neighbors=9, sample_indices=True)
    noisef =ENN(data=data, response=response_trans)
    
    return(noisef)


# # RegENN03Wei3  WEIGHT KNN at rows and features to get MAE

# In[ ]:


################## WEIGHT KNN at rows and features to get MAE.... ######################################


### INPUT
# data=data without response variable
# response= response variable
# alpha= parameter to find noise
# k= number of neihgboors

### OUTPUT
# vector indicating which instances has noisy  in the response variable

import pandas as pd
import numpy as np
from scipy.stats import rankdata
import pandas as pd
import numpy as np
from scipy.stats import rankdata

def RegENN03Wei3(data, response, k=9, alfa=5):
    
    # find position of response variable
    #number=train_data.columns.get_loc(pos)
    
    # Delete the response from train_data
    #data=train_data.loc[:, train_data.columns != number]
    #response=train_data.loc[:,number]
    
    # Feature importance
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(data, response)
    imp=rf.feature_importances_
    
    # Function to get the euclidean distance
    def distan(x,data):
        res=(x-data)**2
        return(np.sum(res,1)**0.5)
    
    # Function to get the euclidean distance
    def distan2(x,data, imp):
        res=(x-data)**2
        res2=res*imp
        return(np.sum(res2,1)**0.5)

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
    distances2=np.apply_along_axis(distan2, 1, data, data, imp)

    # Identify the size position of each value at row
    idxSizeDat=np.apply_along_axis(idxSize, 1, distances)
    idxSizeDat2=np.apply_along_axis(idxSize, 1, distances2)

    # Change the value of the diagonal for the highest number at row in size position data
    np.fill_diagonal(idxSizeDat, idxSizeDat.shape[0]+50)
    np.fill_diagonal(idxSizeDat2, idxSizeDat2.shape[0]+50)

    # Get the index of the smallest values at the row in size position data
    idxF=np.apply_along_axis(idxSmall, 1, idxSizeDat, k)[:,0:k]
    idxF2=np.apply_along_axis(idxSmall, 1, idxSizeDat2, k)[:,0:k]

    # Get mean and standart deviation
    statis=np.apply_along_axis(stat, 1, idxF, response)

    # Get error for each row
    error=np.absolute(statis[:,0]-response)
    #MAE=np.apply_along_axis(stat, 1, idxF2, error)
    #MAE=MAE[:,0]
    
    # Compute MAE weighted
    MAE=[]
    for v in range(distances2.shape[0]):
        dist=distances2[v,idxF2[v,:]]
        dist[dist==0]=0.001
        dato=np.sum(((1/dist)/np.sum(1/dist))*error[idxF2[v,:]])
        MAE.append(dato)
    MAE=np.array(MAE)
    
    
    # Get noise
    # noise=np.absolute(statis[:,0]-response)> ((alfa*MAE[:,1])+MAE[:,0]) # PRUEBA2 CON DESV
    # noise=np.absolute(statis[:,0]-response)> ((alfa*pred[:,1]) # PRUEBA3 CON DESV
    noise=np.absolute(statis[:,0]-response)> (alfa*MAE) 
    noisef=[1.0 if  i == True else 0 for i in noise]
    noisef=np.array(noisef)
    
    # Get error
    # error=np.absolute(statis[:,0]-response)
    
    return noisef


# # DISKR

# In[ ]:


################## DISKR..... ######################################


### INPUT
# data=data without response variable
# response= response variable
# alpha= parameter to find noise
# k= number of neihgboors

### OUTPUT
# vector indicating which instances has noisy  in the response variable


def DISKR(data, response, alfa, k=9):
    
    # Creating replicated files 
    response=np.array(response)
    dataMovil=data.copy()
    responseMovil=response.copy()
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
    
    # Function to get indexes of smallest values. 
    def idxSmall2(x, k): 
        idx = np.argpartition(x,k+1)
        return (idx)

    # Function to get mean and standart deviation.
    def stat(x, response): 
        m = np.mean(response[x])
        s = np.std(response[x])
        return m

    def knn(data,response, k):
        
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
        
        pred=statis
        
        return pred
    
    # Apply knn to predict response
    pred=knn(data=data, response=response, k=9)     
    
    # Get noise
    #PD=np.absolute(statis[:,0]-response)
    PD=np.absolute(pred-response)
    noise=PD > (1-alfa)*response
    noiseIndex =np.where(noise == False)[0]
    noiseIndexOut=np.where(noise == True)[0]
    
    # Delete outliers
    responseMovil=responseMovil[noise != True]
    dataMovil=dataMovil[noise != True]
    PD=PD[noise != True]
    
    # Order data according to PD
    dat=np.concatenate((dataMovil, responseMovil.reshape(-1,1), PD.reshape(-1,1)), axis=1)
    DAT=pd.DataFrame(dat)
    DAT1=DAT.sort_values(by=dat.shape[1]-1, ascending=False)
    DATSORT=np.array(DAT1)
    responseMovil=DATSORT[:,-2]
    dataMovil= DATSORT[:,0:-2 ]
        
    
    # Apply function to get distances of every instance
    distances=np.apply_along_axis(distan, 1, dataMovil, dataMovil)

    # Identify the size position of each value at row
    idxSizeDat=np.apply_along_axis(idxSize, 1, distances)

    # Change the value of the diagonal for the highest number at row in size position data
    np.fill_diagonal(idxSizeDat, idxSizeDat.shape[0]+50)

    # Get the index of the smallest values at the row in size position data
    idxF=np.apply_along_axis(idxSmall, 1, idxSizeDat, k)[:,0:k]
    
    
    # list to save noise vector 
    noisef=[]
    
    for m in range(dataMovil.shape[0]):
        # compute Rbf
        indd=np.where(idxF[:,:] == m)[0]
        if indd.shape[0]!=0:
            preRbf=np.apply_along_axis(stat, 1, idxF[indd,:], responseMovil)
            Rbf= np.sum(preRbf-responseMovil[indd])**2
        
            # compute Raf
            idxSizeDat2=idxSizeDat.copy()
            idxSizeDat2[indd,m]=idxSizeDat.shape[0]+51
            idxF2=np.apply_along_axis(idxSmall, 1, idxSizeDat2[indd,:], k)[:,0:k]
            preRaf=np.apply_along_axis(stat, 1, idxF2, responseMovil)
            Raf= np.sum(preRaf-responseMovil[indd])**2
        
        else:
            Rbf=0
            Raf=0
            

        # Condition to determine if the instance is deleted
        
        if (Raf-Rbf) <= (alfa*Raf):
            noisef.append(1)
            if indd.shape[0]!=0:
                # Rbf for the next
                idxF[indd,:]=idxF2
                idxSizeDat[indd,:]=idxSizeDat2[indd,:]
            
        else:
            noisef.append(0)
    
    # vector of ones equal to the number of outliers instances
    out=np.ones(noiseIndexOut.shape[0])
    
    # concatenate arrays of noise and array of indexes
    noisef2=np.concatenate((np.array(noisef), out), axis=0)
    indexF=np.concatenate((noiseIndex, noiseIndexOut), axis=0)
    
    # sort by indexF
    union=np.concatenate((noisef2.reshape(-1,1), indexF.reshape(-1,1)), axis=1)
    union=pd.DataFrame(union)
    unionF=union.sort_values(by=union.shape[1]-1, ascending=True)
    
    return np.array(unionF[0]) 

