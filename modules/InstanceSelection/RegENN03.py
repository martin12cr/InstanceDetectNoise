

import numpy as np

from sklearn.neighbors import KNeighborsRegressor



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