#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor



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
# vector indicating which instances are noisy in the response variable, 0 means not noisy and 1 means noisy

def RegENN03(data, response, k=9, alpha=0.5, algor='auto', n_jobs=1):
    
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

