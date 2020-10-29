
# coding: utf-8

# In[37]:


# Load
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

############## INPUTS

# data=data as array without response variable, 
# res=response variable as array 
# funcND=function no detect noise (This function should generates a dummie vector to identify which values are noisy). Also it is a lambda function because the data and response are generated inside 
# FuncTrain=Function for the training process 
# pernois=pecentange of instances with noise,  
# split=partions k folds
# type= kind of noise 'uniform' or 'normal'
# filter= If it is equal to 'y' apply instance selection, else not apply 
# rs=random state in cross validation

############## OUTPUTS
# A vector with:  RMSE and MAPE of response prediction, and F-SCORE, PRECISION AND RECALL OF noisy detection, and percentage of cases deleted




def experimentProc(data,response,funcND, funcTrain ,pernois, split=10, type='uniform', filter='y', rs=12, *args): 
    
    # KFold Cross Validation approach
    kf = KFold(n_splits=split,shuffle=True, random_state=rs)

    # Empty vectors to save predictions and real in each fold
    PREAGG=[]
    RECAGG=[]
    F1AGG=[]
    PORAGG=[]
    predAgg= []
    YAgg= []

    # Iterate over each train-test split
    for train_index, test_index in kf.split(data):
    
        # Split train-test
        X_train, X_test = data[train_index], data[test_index]
        Y_train, Y_test = response[train_index], response[test_index]
    
        # Add noise to response
        Y_train, idNoise=AddNoise2(response=Y_train, per=pernois, method=type)    # PER SE DEBE FUNCIONALIZAR
        
        
        
        if filter=='y':
            # Apply instance selection to identify noise
            #idNoisePred =RegENN0(data=X_train, response=Y_trainN, k=9, alfa=5)
              #idNoisePred =func(data=X_train, response=Y_train)
            idNoisePred =f2(X_train,Y_train)
    
            # Delete Noise according to instance selection algorithmn
            X_train= X_train[idNoisePred!=1]
            Y_train=Y_train[idNoisePred!=1]
    
            # Evaluation Noise detection
            PRE=precision_score(idNoise, idNoisePred, pos_label=1)
            REC=recall_score(idNoise, idNoisePred, pos_label=1)
            F1=f1_score(idNoise, idNoisePred, pos_label=1)
            POR=np.sum(idNoisePred)/idNoisePred.shape[0]
            
            # Save evaluation Noise detection
            PREAGG=np.append(PREAGG,PRE)
            RECAGG=np.append(RECAGG,REC)
            F1AGG=np.append(F1AGG,F1)
            PORAGG=np.append(PORAGG,POR)
        
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
    MAPE=np.mean(np.abs(YAgg-predAgg)/YAgg)
    
    
    if filter=='y':
        output=np.array([RMSE,MAPE,np.mean(F1AGG),np.mean(RECAGG), np.mean(PREAGG), np.mean(PORAGG)])
    else:
        output=np.array([RMSE,MAPE,-99,-99,-99,-99])
    
    return (output)

# Example of application 
f2 = lambda data,response: RegENN0(data, response, k=9, alfa=5) # RegENN0 it is one of the noise detection functions 
RESULTADO=experimentProc(data=data1,response=response1,funcND=f2, funcTrain=RandomForestRegressor(max_depth=9, random_state=0), pernois= 0.1)

