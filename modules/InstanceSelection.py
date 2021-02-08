import sklearn

import pandas as pd
import numpy as np


from tqdm import trange

from scipy.stats import rankdata

from sklearn.metrics import f1_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import (NearestNeighbors, KNeighborsRegressor)

from imblearn import under_sampling
from imblearn.under_sampling import EditedNearestNeighbours


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


################## RegENN NOT simultaneouos  ######################################

### INPUT
# data=data without response variable
# response= response variable
# alpha= parameter to find noise
# k= number of neihgboors

### OUTPUT
# vector indicating which instances has noisy  in the response variable
def RegENN01(data, response, k=9, alpha=5):

    k = int(k)
    
    # Creating replicated files 
    response = np.array(response)
    dataMovil = data.copy()
    responseMovil = response.copy()
    cont = 0

    # Vector to save if the instance is noisy
    noisef = np.empty([0])
    indexDel = np.empty([0], dtype=int)
    size = response.shape[0]

    # For each instance
    for i in range(size):
    
        # distances
        distances=distan(data[i,:],dataMovil)   
        
        # change i value for a higher number
        distances[cont]=1000000     
        
        # get index of lower distances
        idxF=np.argpartition(distances, k)[0:k]
        
        # Get mean and standart deviation
        statis=stat(idxF, responseMovil)
        
        # Count of index for data with deleted instances
        cont=cont+1 
        
        # Get noise
        noise=np.absolute(statis[0]-response[i])> (alpha*statis[1])  
       
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



################## RegENN03 alpha* MAE VECINOS..... ######################################

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
def RegENN03(data, response, k=9, alpha=5, algor='auto', n_jobs=1):

    k = int(k)
    
    # get index of neighbors
    neigh = KNeighborsRegressor(n_neighbors=k+1, algorithm=algor, n_jobs=n_jobs).fit(data, response)
    ne=neigh.kneighbors(data, return_distance=False)


    # prediction of response based on neighbors
    pred=[]
    for i in range(response.shape[0]):
        pred.append(np.mean(response[ne[i,1: ]]))
    
    pred=np.array(pred) 
    
    # Get error prediction
    error=np.absolute(pred-response)
    
    MAE=[]
    for i in range(error.shape[0]):
        MAE.append(np.mean(error[ne[i,1: ]]))
    
    MAE=np.array(MAE)
    
    # Get noise
    noise=error > alpha*MAE
    noisef=[1.0 if  i == True else 0 for i in noise]
    noisef=np.array(noisef)

    
    return noisef

################## WEIGHT KNN at rows and features to get MAE.... ######################################

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
def RegENN03Wei3(data, response, k=9, alpha=0.5, algor='auto', n_jobs=1):
    
    k = int(k)
    
    # Feature importance
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(data, response)
    imp=rf.feature_importances_
    imp=imp*data
    imp=imp*data.shape[0]
    
    
    # Apply function to get distances and index neighboors of every instance
    neigh = KNeighborsRegressor(n_neighbors=k+1, algorithm=algor, n_jobs=n_jobs).fit(data, response)
    idxF=neigh.kneighbors(data, return_distance=False)
    
    
    # prediction of response based on neighbors
    pred=[]
    for i in range(response.shape[0]):
        pred.append(np.mean(response[idxF[i,1: ]]))
    
    pred=np.array(pred)
    
    
    #Apply function to get distances WEIGHTED and index neighboors of every instance
    dataTem=imp*data
    neigh = KNeighborsRegressor(n_neighbors=k+1, algorithm=algor, n_jobs=n_jobs).fit(dataTem, response)
    distances2,idxF2=neigh.kneighbors(dataTem, return_distance=True)
    distances2=distances2[:,1:]
    idxF2=idxF2[:,1:]

    # Get error for each row
    error=np.absolute(pred-response)

    # Compute MAE weighted
    MAE=[]
    distances2[distances2==0]=0.001
    for v in range(distances2.shape[0]):
        dist=distances2[v,:]
        dato=np.sum(((1/dist)/np.sum(1/dist))*error[idxF2[v,:]])
        MAE.append(dato)
    MAE=np.array(MAE)
    
    
    noise= error > (alpha*MAE) 
    noisef=[1.0 if  i == True else 0 for i in noise]
    noisef=np.array(noisef)

    
    return noisef



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
def RegBAG(data, response, k=9, alpha=5, numBag=100, samSize=1): 

    k = int(k)
    samSize = int(samSize)
    
    # Size sample bagging
    samSize = samSize * data.shape[0]
    
    # Creating files 
    response = np.array(response)                          # convert response to array 
    noiseMatrix = np.empty([data.shape[0]]).reshape(-1,1)  # creating matriz to save results
    

    # row number of dataframe
    size = response.shape[0]
    
    # Bagging loop
    # for k in range(numBag):
    #for _ in trange(int(numBag)):
    for _ in range(int(numBag)):
        
        # Vector to save noisy instance and the index of noisy instances
        noisef=np.empty([0])                # vector
        indexDel=np.empty([0], dtype=int)   # index
        
        # bagging index selection
        consec = np.r_[0:data.shape[0]]
        r = np.random.choice(a=consec, size=samSize, replace=True)

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

            #  Get the index of the smallest values at the row in size position data
            idxF=idxSmall(idxSizeDat, k)[0:k]

            # Get mean and standart deviation
            statis=stat(idxF, responseMovil)
        
            # Get noise
            noise=np.absolute(statis[0]-response[i])> (alpha*statis[1])  
            if (noise)==True:
                # Detect noise
                noisef=np.append(noisef,1)
            
                # Detect index noise
                indexDel=np.append(indexDel,i)
            
            else:
                noisef=np.append(noisef,0)
            
        # Save matrix    
        noiseMatrix = np.append(noiseMatrix, noisef.reshape(-1,1), axis=1)
    
    # Delete first column
    noiseMatrix=noiseMatrix[:,1:]   
    
    # Creating noise vector 
    noiseF2=np.where(np.sum(noiseMatrix, axis=1) > int(numBag/2), 1, 0)
    
    return (noiseF2)



##################### DiscENN_SIN OPTIMIZADOR ############################

# data=data without response variable
# response= response variable
# strat= method for discretization , we will use uniform
# bins= number of groups to discretize 
# k= number of neihgboors

### OUTPUT
# vector indicating which instances has noisy  in the response variable


def DiscENN(data, response, k=9, kbins_strat='uniform', bins=5): 

    k = int(k)
    bins = int(bins)
    
    # Apply ENN, get indexes of selected samples (No noise samples)
    responseTem=np.array(response).reshape(-1, 1)

    # Apply KBins
    kbins = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=kbins_strat)
    response_trans = kbins.fit_transform(responseTem)

    # Apply ENN for categorical response
    ENN = EditedNearestNeighbours(n_neighbors=k,kind_sel='mode', return_indices=True).fit(data,  response_trans)
    selec=ENN.sample(data,  response_trans)[2]

    # get array of all indexes

    todos=np.arange(response.shape[0])

    # get which were not selected 
    NoSelec=np.setdiff1d(todos, selec, assume_unique=False)

    # add code to selected and not selected 
    unos=np.repeat(0, selec.shape)
    ceros=np.repeat(1, NoSelec.shape)
    noise=np.hstack((unos,ceros))
    indices=np.hstack((selec,NoSelec))
    unidas=np.column_stack((indices,noise))

    # get final noise vector 
    noise = unidas[np.argsort(unidas[:, 0])][:,1]
    
    return noise


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
def DISKR(data, response, k=9, alpha=0.3, algor='brute', n_jobs=1):

    k = int(k)
    
    # Creating replicated files 
    response = np.array(response)
    dataMovil = data.copy()
    responseMovil = response.copy()
    cont=0
    
    # get index of neighbors
    neigh = KNeighborsRegressor(n_neighbors=k+1, algorithm=algor, n_jobs=n_jobs).fit(data, response)
    ne=neigh.kneighbors(data, return_distance=False)


    # prediction of response based on neighbors
    pred=[]
    for i in range(response.shape[0]):
        pred.append(np.mean(response[ne[i,1: ]]))
    
    pred=np.array(pred) 
         
    
    # Get noise
    #PD=np.absolute(statis[:,0]-response)
    PD=np.absolute(pred-response)
    noise=PD > (1-alpha)*response
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
    distances=pdist(dataMovil, 'euclidean')
    distances=squareform(distances)
    
    neigh = NearestNeighbors(n_neighbors=int(dataMovil.shape[0]/2), metric='precomputed')
    neigh.fit(distances)
    distances, idxF=neigh.kneighbors(distances, return_distance=True)
    
    distances=distances[:,1:]
    idxF=idxF[:,1:]
    
    
    # prediction of response based on neighbors
    pred=[]
    for i in range(responseMovil.shape[0]):
        pred.append(np.mean(responseMovil[idxF[i,1:k+1]]))
    
    pred=np.array(pred) 
    preRbf=pred.copy()
   
    # list to save noise vector 
    noisef=[]
    
    # neighboors and distances beetween the kth smallest and substitutes
    distSmall=distances[:,0:k]
    disSust=distances[:, k:]

    idxFSmall=idxF[:,0:k]
    idxFsust=idxF[:,k:]


    # maximum distance
    maxDis=np.max(distances)+1
    
    for m in range(dataMovil.shape[0]):
    #for m in range(24):
        
        #influential points index of each instance
        indd, indd2=np.where(idxFSmall == m)
    
        # minimum Index column substitutes
        indexMin=np.argmin(disSust[indd,:],1) 

        # substitutes indexes 
        susF=idxFsust[indd,indexMin]
        

        if indd.shape[0]!=0:
     
            Rbf= np.sum((preRbf[indd]-responseMovil[indd])**2)
                
            preRaf=((preRbf[indd]*k)+ responseMovil[susF] -responseMovil[m])/k
            Raf=np.sum((preRaf-responseMovil[indd])**2)
   
            print(Rbf, Raf , m)
      
        else:
            Rbf=0
            Raf=1
            
        
        # Condition to determine if the instance is deleted
        
        if (Raf-Rbf) <= (alpha*Raf):
            noisef.append(1)
            if indd.shape[0]!=0:
                # Rbf for the next
                
                # Change in substitutes distances

                fil, col =np.where(idxFsust == m)
                disSust[fil,col]=maxDis

                # Change in substitutes idxFSmall 
                idxFSmall[indd,indd2]=susF
                
                
                preRbf[indd]=preRaf
            
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

