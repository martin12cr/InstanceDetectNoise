
# coding: utf-8

# #  IMPORT MODULES

# In[ ]:


import pandas as pd
import sys
# sys.path.append('/home/msolis/InstanceDetectNoise/modules/') # path in KABRE 
sys.path.append('D:/CIENCIA DE DATOS PROYECTOS/noise in regression/modules/')


import IS
from IS2 import RegENN01
from IS2 import RegENN03
from IS2 import RegBAG
from IS2 import DiscENN
from IS2 import ENN
import ExperimentF


# # PATH AND PARAMETERS 

# In[91]:


import numpy as np

############################## PATH DATA FRAMES ########################################

# RootOr='/work/msolis/data/'  # PATH KABRE 
RootOr='D:/CIENCIA DE DATOS PROYECTOS/noise in regression/prueba/'

############################### NOISE PARAMETERS #######################################
# PARAMETROS PARA PRUEBAS
#array_1 = np.array([0.1, 0.3])  # instances noise
#array_2 = np.array([0.1, 0.5]) # magnitud noise

array_1 = np.array([0.1, 0.3, 0.5])  # instances noise
array_2 = np.array([0.1, 0.5, 1, 2]) # magnitud noise
array_3 = np.array([1]) # filter, 1 means yes
arrNoise=np.array(np.meshgrid(array_1, array_2, array_3)).T.reshape(-1, 3) # matriz noise
b1=np.array([0,0,1]).reshape(1,-1) # adding 0 noise to matrix
arrNoise=np.append(arrNoise, b1, axis=0)  # adding 0 noise to matrix
#b2=np.array([0,0,0]).reshape(1,-1) # adding 0 noise to matrix
#arrNoise=np.append(arrNoise, b2, axis=0)


################################# FUNCTION iNSTANCE SELECTION AND PARAMETERS #############


j=0.5  # Parameter of function. For RegENN01,RegENN01, and RegBAG we also will prove changing j to 1 , 3 and 5 
f2=lambda data,response: RegENN01(data, response, alfa=j, k=9)  # function instance selection
numberData=50    # number of data to include. The final number is 50
met='RegENN01_' # name of the instance selection algorithm 

#f2= lambda data,response:RegENN03(data, response,alfa=j, k=9, numBag=100, samSize=1)
#f2= lambda data,response:RegBAG(data, response,alfa=j, k=9, numBag=100, samSize=1)
#f2= lambda data,response: DiscENN(data, response, strat='uniform', bins=j,  k=9)  # In this case we will prove j with:2,3,4,5



# Creating functions manually
#f1 = lambda data,response: RegENN01.RegENN01(data, response, alfa=0.5, k=9)
#f2 = lambda data,response: RegENN01.RegENN01(data, response, alfa=1.0, k=9)
#f3 = lambda data,response: RegENN01.RegENN01(data, response, alfa=2, k=9)
#listFun=[f1,f2, f3]


# # FINAL EXECUTION

# In[92]:


# FUNCTION FOR FINA EXECUTION
def final(numberData, met, j, *args):
    
    # Applying process
    matriz2=pd.DataFrame()  # Empty data frame to concatenate
    cont=0   
    matriz=Experiment.proceso(func=f2, ParNoise=arrNoise, Root=RootOr, numData=numberData) 
    
    # Add name of method and its paramter to dataset of results
    matriz=matriz.assign(method= met, param=j)

    # name to save and path
    name=met+str(float(j))+'.csv'
    nameF='D:/CIENCIA DE DATOS PROYECTOS/noise in regression/modules/'+name
    #nameF='/home/msolis/InstanceDetectNoise/results/'+name  # PATH KABRE 
    
    
    # export dataset
    matriz.to_csv(nameF)
    
    return(matriz)

# EXECUTING FUNCTION
matriz2=final(numberData, met, j, f2, arrNoise, RootOr)


# # It will not uset. For evaluation all the Instance selection parameters simultaneously 

# In[44]:


parIS = np.array([0.5,  1, 2, 4])
parIS=parIS.reshape(-1,1) # only if there is one row
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


# In[86]:


#matriz5=Experiment.procesoRep(funcF=listFun, arrNoise=arrNoise, RootOr=RootOr, numberData=1, parFuncNoise=parIS, met='RegENN01')
import pandas as pd
numberData=1
met=['RegENN01','DiscENN']
met=['RegENN01']
matriz2=pd.DataFrame()  # Empty data frame to concatenate
contList=0
for k in lista:
    cont=0
    for j in parIS[:,0]:
    
        # Appling function for process
        f2=k
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
        matriz= matriz.assign(method= met[contList])
        # concatenate dataframes of each iteration
        matriz2=pd.concat([matriz2, matriz], sort=False)
        
        # Add name of method
        cont=cont+1
    contList=contList+1
    

