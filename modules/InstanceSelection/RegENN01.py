
import numpy as np

from .utils import (distan, stat)


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