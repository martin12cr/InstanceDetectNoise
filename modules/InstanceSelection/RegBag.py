

import numpy as np

from .utils import *


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
def RegBag(data, response, k=9, alpha=5, numBag=100, samSize=1): 

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
