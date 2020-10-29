
# coding: utf-8

# In[27]:


import sys
sys.path.append('/home/msolis/InstanceDetectNoise/modules/')
import IS
from IS import RegENN01
from ENN import ENN
import Experiment


# In[28]:


import numpy as np


# PATH
RootOr='/work/msolis/data/'
# Creating noise matrix


######## NOISE PARAMETERS 
# PARAMETROS PARA PRUEBAS
#array_1 = np.array([0.1, 0.3])  # instances noise
#array_2 = np.array([0.1, 0.5]) # magnitud noise

array_1 = np.array([0.1, 0.3, 0.5])  # instances noise
array_2 = np.array([0.1, 0.5, 1, 2]) # magnitud noise
array_3 = np.array([1]) # filter, 1 means yes
arrNoise=np.array(np.meshgrid(array_1, array_2, array_3)).T.reshape(-1, 3) # matriz noise
b1=np.array([0,0,1]).reshape(1,-1) # adding 0 noise to matrix
#b2=np.array([0,0,0]).reshape(1,-1) # adding 0 noise to matrix
arrNoise=np.append(arrNoise, b1, axis=0)  # adding 0 noise to matrix
#arrNoise=np.append(arrNoise, b2, axis=0)


######## PARAMETERS AND FUNCTIONS FOR REPLICATIONS
parIS = np.array([0.5,  1, 2])
parIS=parIS.reshape(-1,1) # only if there is one row

#parIS = np.array([2,  3, 4])
#parIS=parIS.reshape(-1,1) # only if there is one row


# Creating functions manually
#f1 = lambda data,response: RegENN01.RegENN01(data, response, alfa=0.5, k=9)
#f2 = lambda data,response: RegENN01.RegENN01(data, response, alfa=1.0, k=9)
#f3 = lambda data,response: RegENN01.RegENN01(data, response, alfa=2, k=9)
#listFun=[f1,f2, f3]


# In[44]:


#matriz5=Experiment.procesoRep(funcF=listFun, arrNoise=arrNoise, RootOr=RootOr, numberData=1, parFuncNoise=parIS, met='RegENN01')
import pandas as pd
numberData=1
met='RegENN01'
matriz2=pd.DataFrame()  # Empty data frame to concatenate
cont=0

for j in parIS[:,0]:
    
    # Appling function for process
    f2=lambda data,response: RegENN01(data, response, alfa=j, k=9)
    #f2=lambda data,response:IS.DiscENN(data, response, strat='uniform', bins=j,  k=9)
    #f2= lambda data,response:IS.RegBAG(data, response,alfa=j, k=9, numBag=100, samSize=1)
    matriz=Experiment.proceso(func=f2, ParNoise=arrNoise, Root=RootOr, numData=numberData) 
    
    # Generate a vector to save the IS parameters
    rep=parIS[cont,:]  # chose the value of parameters
    rep=rep.reshape(1,-1)  # convert to 2Darray
    new=np.repeat(rep, matriz.shape[0], axis=0)  # repeat row n times

    # Join vector (array) with data frame 
    new=pd.DataFrame(new)  # convert to dataframe
    new.index += 1         # change the index to join data frames
    matriz=matriz.join(pd.DataFrame(new))  # joint array or vector of parameters with dataframe
      
    # concatenate dataframes of each iteration
    matriz2=pd.concat([matriz2, matriz], sort=False)
        
    # Add name of method
    matriz2=matriz2.assign(method= met)
    cont=cont+1
    
    matriz2.to_csv('/home/msolis/InstanceDetectNoise/results/')


