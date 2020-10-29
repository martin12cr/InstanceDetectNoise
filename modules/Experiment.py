
# coding: utf-8

# # Function to load DATA FRAMES

# In[2]:


###################### Function to load DATAs ###############################
# dirRoot = path where data are located
# posDat = Data position 
import pandas as pd

def open (dirRoot, posDat):
    import os
    listaDir=[]
    listaNames=os.listdir(dirRoot)
    for i in range(len(listaNames)):
        text=dirRoot+listaNames[i]
        listaDir.append(text)
       
    dir=listaDir[posDat]
    nameData=listaNames[posDat]
    train_data = pd.read_csv(dir,sep=",", header=0)
    name=list(range(train_data.shape[1]))
    train_data = pd.read_csv(dir,sep=",", names=name)
    return train_data,nameData


# # Function to load DATA FRAMES

# In[3]:


######### Function to data preparation(e.g, normalization, response separation, etc)

# dataFrame=dataframe 
# pos=column position of response in dataframe
import numpy as np
def file_preparation(dataFrame,pos): 
    
    # Sort dataFrame
    dataFrame=dataFrame.sample(frac=1)  
    
    # Normalize function
    def norm(x):
        new=(x-np.min(x))/(np.max(x)-np.min(x))
        return(new)
    
    # Exclude the response from train_data
    number=pos  
    data=dataFrame.loc[:, dataFrame.columns != number]
    response=dataFrame.loc[:,number]
    
    # applying normalize function to every column 
    data=np.apply_along_axis(norm, 0, data.values)
    response=np.array(response)
    return(data,response)


# data1,response1=file_preparation(train_data2,train_data2.shape[1]-1)


# # NOISE FUNCTION 

# In[4]:


import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn import preprocessing
import os

# NOISE FUNCTION 
#### inputs 
#response = response variable or output variable as vector
# per= percentage of noise instances
# method=noise method, 'normal (gaussian)' or 'uniform'

### Outputs
# return two arrays, one with the response variable with noise in some instances, and a dummy vector indicating if the instance is noisy


def AddNoise2(response, per, method, mag=0.3,rs=30):    
    
    np.random.seed(rs)
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
    elif (method=='uniform'):
        noise1=np.random.uniform(np.min(response),np.max(response),c1)*mul1
    elif (method=='persize'):
        if per >0:
            noise1= arr[:,2][-c1:response.shape[0]]*mag*mul1
        else:
            noise1=np.zeros(response.shape[0])
            
    if (per>0):    
        # Stack noise vectors
        noise=np.hstack((np.zeros(response.shape[0]-(c1)),noise1))
    else:
        noise=noise1
    
    # stack noise vector in matrix
    arr=np.column_stack((arr,noise))
    
    # Add noise to the response
    #arr[:,2]=arr[:,2]+arr[:,3]
    agg=arr[:,2]+arr[:,3]
    arr=np.column_stack((arr,agg))
    
    # Sort array according to consecutive
    arr=arr[np.argsort(arr[:,0])]
    
    # Recoding noise in two groups, 1= noise instance, 0= not noise
    noiseAbs=np.abs(arr[:,3])
    noiseAbs[noiseAbs == 0] = 0
    noiseAbs[noiseAbs >0] = 1
    noiseAbs[noiseAbs <0] = 1
    
    return([arr[:,4], noiseAbs])
    #return([arr, noiseAbs])


# # Function for experimental process

# In[5]:


########################## Function for experimental process #####################


############## INPUTS

# data=data as array without response variable, 
# res=response variable as array 
# funcND=function no detect noise (This function should generates a dummie vector to identify which values are noisy). Also it is a lambda function because the data and response are generated inside 
# FuncTrain=Function for the training process 
# pernois=pecentange of instances with noise,  
# split=partions k folds
# type= kind of noise 'uniform' or 'normal'
# filter= If it is equal to 1 apply instance selection, 0 not apply 
# rs=random state in cross validation

############## OUTPUTS
# A vector with:  RMSE and MAPE of response prediction, and F-SCORE, PRECISION AND RECALL OF noisy detection, and percentage of cases deleted

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


def experimentProc(data,response,funcND, funcTrain ,pernois, split=10, type='uniform', magn=0.3, filter=1, rs=30, *args): 
    from sklearn.metrics import r2_score
    # KFold Cross Validation approach
    kf = KFold(n_splits=split,shuffle=True, random_state=rs)

    # Empty vectors to save predictions and real in each fold
    PREAGG=[]
    RECAGG=[]
    F1AGG=[]
    NUMAGG=[]
    predAgg= []
    YAgg= []

    # Iterate over each train-test split
    for train_index, test_index in kf.split(data):
    
        # Split train-test
        X_train, X_test = data[train_index], data[test_index]
        Y_train, Y_test = response[train_index], response[test_index]
    
        # Add noise to response
        Y_train, idNoise=AddNoise2(response=Y_train, per=pernois, method=type, mag=magn)    # PER SE DEBE FUNCIONALIZAR
        
        
        
        if filter==1:
            # Apply instance selection to identify noise
            #idNoisePred =RegENN0(data=X_train, response=Y_trainN, k=9, alfa=5)
              #idNoisePred =func(data=X_train, response=Y_train)
            idNoisePred =funcND(X_train,Y_train)
    
            # Delete Noise according to instance selection algorithmn
            X_train= X_train[idNoisePred!=1]
            Y_train=Y_train[idNoisePred!=1]
    
    
            # Evaluation Noise detection
            PRE=precision_score(idNoise, idNoisePred, pos_label=1, zero_division=0)
            REC=recall_score(idNoise, idNoisePred, pos_label=1, zero_division=0)
            F1=f1_score(idNoise, idNoisePred, pos_label=1, zero_division=0)
            NUM=np.sum(idNoisePred)
            
            # Save evaluation Noise detection
            PREAGG=np.append(PREAGG,PRE)
            RECAGG=np.append(RECAGG,REC)
            F1AGG=np.append(F1AGG,F1)
            NUMAGG=np.append(NUMAGG,NUM)
        
        #Train the model
        #regr=RandomForestRegressor(max_depth=9, random_state=0)
        regr = funcTrain
        model = regr.fit(X_train, Y_train)
    
        # Generate prediction
        pred=model.predict(X_test)
    
        # Save real and prediction vectors of the  fold 
        predAgg=np.concatenate((predAgg,pred), axis=0)
        YAgg=np.concatenate((YAgg,Y_test), axis=0)

    
    # Evaluation
    RMSE=np.sqrt(mean_squared_error(YAgg, predAgg))
    MAPE=np.mean(np.abs(YAgg-predAgg)/(YAgg+0.001))
    correlation_matrix=np.corrcoef(YAgg,predAgg)
    correlation_xy = correlation_matrix[0,1]
    R2=correlation_xy**2
    POR=np.sum(NUMAGG)/data.shape[0]
    if filter==1:
        output=np.array([RMSE,MAPE,R2,POR,np.mean(F1AGG),np.mean(RECAGG), np.mean(PREAGG)])
    else:
        output=np.array([RMSE,MAPE,R2,-99,-99,-99,-99])
    
    return (output)


# # Function for process

# In[6]:


######### Function to generate the experimental process ##################

# func=lamba function with the parameters of Noise Selection algorithmn 
# ParNoise = array with values of noise percentaje, magnitud noise, and filer application (1=apply filter)
# Root = path where data are located
# numData = number of data to load

'''
# Path
RootOr='D:/CIENCIA DE DATOS PROYECTOS/noise in regression/prueba/'
# Creating noise matrix
array_1 = np.array([0.1, 0.3, 0.5])  # instances noise
array_2 = np.array([0.1, 0.5, 1, 2]) # magnitud noise
array_3 = np.array([0, 1]) # filter, 1 means yes
arrNoise=np.array(np.meshgrid(array_1, array_2, array_3)).T.reshape(-1, 3) # matriz noise
b1=np.array([0,0,0]).reshape(1,-1) # adding 0 noise to matrix
b2=np.array([0,0,1]).reshape(1,-1) # adding 0 noise to matrix
arrNoise=np.append(arrNoise, b1, axis=0)  # adding 0 noise to matrix
arrNoise=np.append(arrNoise, b2, axis=0)
# creating function 
f2 = lambda data,response: RegENN01(data, response, k=9, alfa=4)
'''
 

def proceso (func, ParNoise, Root, numData=2):
    DATAF=pd.DataFrame(columns=['pernois', 'magnois', 'filter','RMSE','MAPE', 'R2', 'F1', 'REC', 'PRE', 'POR', 'DATA'])
    cont=0
    for pos in range(numData):   #este parámetro numData se puede cambiar para que quede automatizado 
        for i in range(ParNoise.shape[0]):
            trainData, nameData=open(dirRoot=Root, posDat=pos) # function to load data
            data1,response1=file_preparation(trainData,trainData.shape[1]-1) # Function to file preparation
            RESULTADO=experimentProc(data=data1,response=response1,funcND=func, funcTrain=RandomForestRegressor(max_depth=9, random_state=0), pernois= ParNoise[i,0],type='persize', magn=ParNoise[i,1], filter=int(ParNoise[i,2]))
            cont=cont+1
            rowAdd=list(np.append(ParNoise[i,:],RESULTADO))
            rowAdd.append(nameData)
            DATAF.loc[cont]=rowAdd      
    return DATAF 


# # Function for process- larger

# In[7]:


######### Function to generate the experimental process ##################
######## for each experimental parameter of IS algorithm #################


# funcF=lamba function with the parameters of Noise Selection algorithmn 
# arrNoise= array with values of noise percentaje, magnitud noise, and filer application (1=apply filter)
# RootOr = path where data are located
# numberData = number of data to load
# parFuncNoise= array with parameters values of IS algorithm 
# met=name of algorithm used 

'''
# Creating the function automatically, but it generate problems, maybe bug
listFun=[]
li=(7, 11, 12)
for j in li:
    f2=lambda data,response: RegENN01(data, response, alfa=j, k=9)
    listFun.append(f2)

# path
RootOr='D:/CIENCIA DE DATOS PROYECTOS/noise in regression/prueba/'

# Paramters
parIS = np.array([0.5,  1.0, 2])
parIS=parIS.reshape(-1,1) # only if there is one row

# Creating functions manually
f1 = lambda data,response: RegENN01(data, response, alfa=0.5, k=9)
f2 = lambda data,response: RegENN01(data, response, alfa=1.0, k=9)
f3 = lambda data,response: RegENN01(data, response, alfa=2, k=9)
listFun=[f1,f2, f3]

'''

def procesoRep(numberData,funcF, arrNoise, RootOr, parFuncNoise, met='RegENN01'):
    matriz2=pd.DataFrame()  # Empty data frame to concatenate
    #for j in range(parFuncNoise.shape[0]):
    cont=0   
    for j in funcF:
        # Appling function for process
        matriz=proceso(func=j, ParNoise=arrNoise, Root=RootOr, numData=numberData) 
        # Generate a vector to save the IS parameters
        rep=parFuncNoise[cont,:]  # chose the value of parameters
        rep=rep.reshape(1,-1)  # convert to 2Darray
        new=np.repeat(rep, matriz.shape[0], axis=0)  # repeat row n times

        # Join vector (array) with data frame 
        new=pd.DataFrame(new)  # convert to dataframe
        new.index += 1         # change the index to join data frames
        matriz=matriz.join(pd.DataFrame(new))  # joint array or vector of parameters with dataframe
    
        # concatenate dataframes of each iteration
        matriz2=pd.concat([matriz2, matriz])
        
        cont=cont+1
        # Add name of method
    matriz2=matriz2.assign(method= met)
        
    return(matriz2)

