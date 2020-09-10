
# coding: utf-8

# In[ ]:


# WITH NEURONAL NETWORK 
### INPUT
# data=data without response variable
# response= response variable
# alpha= parameter to find noise
# k= number of neihgboors

### OUTPUT
# vector indicating which instances has noisy  in the response variable


def RegENN(data, response, k, alfa):
    
    # find position of response variable
    #number=train_data.columns.get_loc(pos)
    
    # Delete the response from train_data
    #data=train_data.loc[:, train_data.columns != number]
    #response=train_data.loc[:,number]
    
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

    # Function to get standart deviation.
    def stat(x, response): 
        s = np.std(response[x])
        return (s)
    
    # Function to generate models 
    def model(x,data, response):
        from sklearn.neural_network import MLPRegressor
        idd=np.where(np.all(data==x,axis=1))
        idd=idd[0].item()
        responseT=np.delete(arr=np.array(response), obj=idd, axis=0)
        dataT=np.delete(arr=data, obj=idd, axis=0)  
        Model=MLPRegressor(hidden_layer_sizes=(int((dataT.shape[1]+1)/2), ), learning_rate_init=0.3,random_state=1, momentum=0.2).fit(dataT, responseT)
        ypred=Model.predict(np.reshape(x, (1, x.shape[0])))
        return(ypred)
 
    
    # Apply algorithm 
    ypred=np.apply_along_axis(model, 1, data, data, response)
    ypred=np.reshape(ypred, (ypred.shape[0], ))
   
    # Apply function to get distances of every instance
    distances=np.apply_along_axis(distan, 1, data, data)

    # Identify the size position of each value at row
    idxSizeDat=np.apply_along_axis(idxSize, 1, distances)

    # Change the value of the diagonal for the highest number at row in size position data
    np.fill_diagonal(idxSizeDat, idxSizeDat.shape[0]+50)

    # Get the index of the smallest values at the row in size position data
    idxF=np.apply_along_axis(idxSmall, 1, idxSizeDat, k)[:,0:k]

    # Get mean and standart deviation
    statis=np.apply_along_axis(stat, 1, idxF, response)

    # Get noise
    noise=  np.absolute(ypred-response) > (alfa*statis)  
    noisef=[1.0 if  i == True else 0 for i in noise]
    noisef=np.array(noisef)
    
    return(noisef)

