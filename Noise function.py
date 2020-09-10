
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn import preprocessing
import os

# NOISE FUNCTION 2
#### inputs 
#response = response variable or output variable as vector
# per= percentage of noise instances
# method=noise method, 'normal (gaussian)' or 'uniform'

### Outputs
# return two arrays, one with the response variable with noise in some instances, and a dummy vector indicating if the instance is noisy


def AddNoise2(response, per, method):    
    # Create array with consecutive, random vector and response vector
    arr=np.stack((np.r_[1:response.shape[0]+1], np.random.rand(response.shape[0]), response), axis=-1)

    # Sort array according to random vector 
    arr=arr[np.argsort(arr[:,1])]

    #### Generate noise vector

    # number of values
    c1=int(response.shape[0]*per)

    # to make the ramdom noise positive or negative
    mul1=np.random.randint(2, size=c1)
    mul1[mul1== 0] = -1
    
    if (method=='normal'):
        # to make the ramdom noise positive or negative
        noise1 = np.random.normal(np.mean(response),np.std(response),c1)*mul1
    else:
        noise1=np.random.uniform(np.min(response),np.max(response),c1)*mul1
    
    
    # Stack noise vectors
    noise=np.hstack((np.zeros(response.shape[0]-(c1)),noise1))
   
    # stack noise vector in matrix
    arr=np.column_stack((arr,noise))
    
    # Add noise to the response
    arr[:,2]=arr[:,2]+arr[:,3]
    
    # Sort array according to consecutive
    arr=arr[np.argsort(arr[:,0])]

  
    # Recoding noise in two groups, 1= noise instance, 0= not noise
    noiseAbs=np.abs(arr[:,3])
    noiseAbs[noiseAbs == 0] = 0
    noiseAbs[noiseAbs >0] = 1
    noiseAbs[noiseAbs <0] = 1
    
    return([arr[:,2], noiseAbs])

