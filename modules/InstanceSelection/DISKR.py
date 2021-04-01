
import numpy as np
from numpy.lib import utils
import pandas as pd

from .utils import *

from sklearn.neighbors import KNeighborsRegressor



################## DISKR..... ######################################


### INPUT
# data=data without response variable
# response= response variable
# alpha= parameter to find noise
# k= number of neihgboors
# algor=auto or brute, brute it is better for larger datasets 
# n_jobs= The number of parallel jobs to run for neighbors search. None means 1 unless in a joblib.parallel_backend context. 
#-1 means using all processors. See Glossary for more details. Doesn’t affect fit method.


### OUTPUT
# vector indicating which instances has noisy  in the response variable
def DISKR(data, response, k=9, alpha=0.3, algor='auto'):
    
    # Creating replicated files 
    dataMovil=data.copy()
    responseMovil=response.copy()
    k=int(k)
    cont=0    
        
    # get index of neighbors
    neigh = KNeighborsRegressor(n_neighbors=k+1, algorithm=algor, metric='euclidean').fit(dataMovil, responseMovil)
    ne=neigh.kneighbors(dataMovil, return_distance=False)


    # prediction of response based on neighbors
    pred=[]
    for i in range(responseMovil.shape[0]):
        pred.append(np.mean(responseMovil[ne[i,1: ]]))
    
    pred=np.array(pred) 
            
    # Get noise
    #PD=np.absolute(statis[:,0]-response)
    PD=np.absolute(pred-responseMovil)
    noise=PD > (1-alpha)*responseMovil
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
    distances=pdist (dataMovil, 'euclidean')
    distances=squareform(distances)
    

    # Identify the size position of each value at row
    idxSizeDat=np.zeros([distances.shape[0],distances.shape[1]])

    for i in range(dataMovil.shape[0]):
        idxSizeDat[i,:]=rankdata(distances[i,:], method='ordinal')

    # Change the value of the diagonal for the highest number at row in size position data
    np.fill_diagonal(idxSizeDat, idxSizeDat.shape[0]+50)

    # Get the index of the smallest values at the row in size position data
    idxF=np.zeros([idxSizeDat.shape[0],k])

    for i in range(idxSizeDat.shape[0]):
        idxF[i,:]=np.argpartition(idxSizeDat[i,:], k)[0:k]
    
    idxF=idxF.astype(int)
    # list to save noise vector 
    noisef=[]
    
   
    # Creating for prediction     
    for m in range(dataMovil.shape[0]):
    #for m in range(2):
    # compute Rbf
        indd=np.where(idxF[:,:] == m)[0]
        if indd.shape[0]!=0:
            preRbf=[]
            for i in range(len(indd)):
                preRbf.append(np.mean(responseMovil[idxF[indd[i],:]]))
            preRbf=np.array(preRbf)  
            Rbf= np.sum((preRbf-responseMovil[indd])**2)
    
    
            # compute Raf
            idxSizeDat2=idxSizeDat.copy()
            idxSizeDat2[indd,m]=idxSizeDat.shape[0]+51
            preRaf=[]

            idxF2=np.zeros([indd.shape[0],k]).astype(int)
            for i in range(len(indd)): 
                idxF2[i,:]=np.argpartition(idxSizeDat2[indd[i],:], k)[0:k]
                preRaf.append(np.mean(responseMovil[idxF2[i,:]]))

            preRaf=np.array(preRaf)  
            Raf= np.sum((preRaf-responseMovil[indd])**2)
    
        else:
            Rbf=0
            Raf=10
                         
        if (Raf-Rbf) <= (alpha*Raf):
            noisef.append(1)
            # Rbf for the next
            idxF[m,:]=-1
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
        
    return unionF[0]   