import sys, time

import numpy as np
import pandas as pd

from tqdm import tqdm
from .istarmap import *
from multiprocessing import Pool

from .entities import Dataset
from .data_manager import preprocessing

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error




# # NOISE FUNCTION 


# NOISE FUNCTION 
#### inputs 


### Outputs
# return two arrays, one with the target variable with noise in some instances, and a dummy vector indicating if the instance is noisy
"""
    Function designed to add percentage of noise to a fixed percentage of the given samples

    target:     Target variable or output variable as vector
    perc:       Percentage of noisy instances
    method:     Noise method, can be 'normal (gaussian)', 'uniform' or 'persize'

    return:
    Two arrays, one with the target variable with noise in some instances, and a dummy vector indicating if the instance is noisy

"""
def add_noise(target, perc, method, mag=0.3, rs=30):    
    

    # First validate if wee need to add noise
    if(perc > 0): 

        # Set the random seed for reproduceability
        np.random.seed(rs)

        # Number of noisy instances
        num_noisy_inst = int(target.shape[0] * perc)
        # Target length
        target_len = len(target)

        # Generate vector to encode noise addition
        noisy_instances = np.random.randint(2, size=num_noisy_inst)
        # If 1 the noise is added to the value, if 0 the noise is substracted
        noisy_instances[noisy_instances == 0] = -1
        # Add zeros encoding clean instances
        noisy_instances = np.hstack((np.zeros(target_len - num_noisy_inst), noisy_instances))
        # Shuffle noisy instances
        np.random.shuffle(noisy_instances)
    
        if (method == 'normal'):
            # to make the ramdom noise positive or negative
            #noise = np.random.normal(np.mean(target), np.std(target), num_noisy_inst) * noisy_instances
            sys.exit("Support for normal noise addition is not available")

        elif (method == 'uniform'):
            #noise = np.random.uniform(np.min(target), np.max(target), num_noisy_inst) * noisy_instances
            sys.exit("Support for uniform noise addition is not available")

        # For noise thats relative to the original value
        elif (method == 'persize'):
            # Calc the noise to be added as a msg of the original value 
            noise = target * mag * noisy_instances
       
        # Add noise to the target
        noisy_target = target + noise
        
        # Encode the noisy instance, if 0 the the intance is clean 
        # else, it isnt
        noisy_instances[noisy_instances != 0] = 1

        # arr[0] = Numers from 0 to len(target)
        # arr[1] = Shuffle vector
        # arr[2] = Target
        # arr[3] = Noise vector
        # arr[4] = Noisy target

        return ([noisy_target, noisy_instances])


    # If perc is zero then no noise is needed
    else:
        # No noise means noisy intances vector is zero and 
        # the noisy target is the target 
        return ([target, np.zeros(target.shape[0])])


"""
    Function to excecute a specific noise combination on a given dataset

    dataset:        Dataset class that contains the data and target values
    is_algorithm:   Instance selection algorithm, must be callable 
    aux_algoritm:   Auxiliar algorithm to be trained and used for the instance selection process, must be callable
    noise_perc:     Pecentange of instances with noise
    noise_magn:     Percentage of the amount noise on corrupted instances
    noise_type:     Defines the noise to be used, can be 'normal (gaussian)', 'uniform' or 'persize'
    split:          Number of folds used for cross validation
    filtr:         If it is equal to 1 apply instance selection, 0 not apply 
    rs:             Random state in cross validation

    return:
    A vector with RMSE and MAPE of target prediction, and F-SCORE, PRECISION AND RECALL OF noisy detection, and percentage of cases deleted
"""
def experiment_on_noise(dataset, is_algorithm, aux_algoritm, noise_perc, noise_magn=0.3, noise_type='persize', filtr=1, split=10,  rs=30): 

    # KFold Cross Validation approach
    kf = KFold(n_splits=split, shuffle=True, random_state=rs)

    # Empty vectors to save predictions and real in each fold
    PREAGG=[]
    RECAGG=[]
    F1AGG=[]
    NUMAGG=[]
    predAgg= []
    YAgg= []

    # Iterate over each train-test split
    for train_index, test_index in kf.split(dataset.x):
    
        # Split train-test
        X_train, X_test = dataset.x[train_index], dataset.x[test_index]
        Y_train, Y_test = dataset.y[train_index], dataset.y[test_index]
    
        # Add noise to target
        Y_train, idNoise = add_noise(target=Y_train, perc=noise_perc, method=noise_type, mag=noise_magn)    
        
        
        
        if(filtr == 1):
            # Apply instance selection algorithm
            idNoisePred = is_algorithm.evaluate(X_train,Y_train)
    
            # Delete Noise according to instance selection algorithmn
            X_train = X_train[idNoisePred != 1]
            Y_train = Y_train[idNoisePred != 1]
    
            # Evaluation Noise detection
            PRE = precision_score(idNoise, idNoisePred, pos_label=1, zero_division=0)
            REC = recall_score(idNoise, idNoisePred, pos_label=1, zero_division=0)
            F1 = f1_score(idNoise, idNoisePred, pos_label=1, zero_division=0)
            NUM = np.sum(idNoisePred)
            
            # Save evaluation Noise detection
            PREAGG = np.append(PREAGG, PRE)
            RECAGG = np.append(RECAGG, REC)
            NUMAGG = np.append(NUMAGG, NUM)
            F1AGG = np.append(F1AGG, F1)
        
        # Train the auxiliar algorithm
        aux_algoritm = aux_algoritm.fit(X_train, Y_train)
    
        # Generate prediction
        pred = aux_algoritm.predict(X_test)
    
        # Save real and prediction vectors of the  fold 
        predAgg = np.concatenate((predAgg, pred), axis=0)
        YAgg = np.concatenate((YAgg, Y_test), axis=0)

    
    # Evaluation
    RMSE = np.sqrt(mean_squared_error(YAgg, predAgg))
    MAPE = np.mean(np.abs(YAgg - predAgg) / (YAgg + 0.001))
    correlation_matrix = np.corrcoef(YAgg, predAgg)
    correlation_xy = correlation_matrix[0, 1]
    R2 = correlation_xy ** 2
    POR = np.sum(NUMAGG) / dataset.x.shape[0]
    if filtr == 1:
        output = np.array([RMSE, MAPE, R2, POR, np.mean(F1AGG), np.mean(RECAGG), np.mean(PREAGG)])
    else:
        output = np.array([RMSE, MAPE, R2,- 99, -99, -99, -99])
    
    return np.append(np.append(np.array([noise_perc, noise_magn, noise_type, filtr]) , output), [dataset.name, is_algorithm.name])


"""

    Function to evaluate a set of algorithms on a given dataset

    alg:            Dictionary containing the lambda function of the instance selection algorithms to evaluate
    data_path:      Path to the CSV file we'll work on
    noise_params:   
    
"""
def experiment_on_data(algorithms, data_path, noise_params):

    # Results container 
    results_df = pd.DataFrame(columns=["noise_perc", "noise_magn", "noise_type", "filter","RMSE","MAPE", "R2", "F1", "REC", "PRE", "POR", "DATA", "ALG"])

    # Get the data name
    name_data = data_path.split('/')[-1].split('.')[0]

    # Load dataset
    dataset = pd.read_csv(data_path, sep=",", header=None)
    # Normalize columns for x and y data
    dataset = preprocessing(dataset, dataset.shape[1] - 1)
    # Get the data and targets into a container to generate the meshgrid
    dataset = Dataset(name_data, dataset[0], dataset[1])

    # Get time stamps
    #start_time = time.time()

    # Matrix of combined parameters for the experiments
    exp_params = np.array(np.meshgrid(  dataset,
                                        algorithms, 
                                        RandomForestRegressor(max_depth=9, random_state=0), 
                                        noise_params[0], # Number of added noisy instances (%)
                                        noise_params[1], # Percentage of noise added over the instance
                                        noise_params[2], # Noise type 
                                        noise_params[3]  # Filter 1 means yes
                                        )).T.reshape(-1, 7)

    
    # Start with the number of processors
    with Pool() as pool:
        # Run experiments in parallel and store results
        for noisy_result in  tqdm(pool.istarmap(experiment_on_noise, exp_params),
                                total=len(exp_params), desc="Experimenting on " + name_data):

            # Add the results to the dataframe
            results_df.loc[len(results_df)] = noisy_result

    #exec_time = time.time() - start_time
    #print("Finished experimenting on " + name_data + " dataset in " + str(int(exec_time // 60)) + ":" + str(int(exec_time % 60))) 

    # Reset the index in case addigng the rows caused indexing errors
    return results_df.reset_index(drop=True) 



def one_test(algorithms, data_path, noise_params):

    # Results container 
    results_df = pd.DataFrame(columns=["noise_perc", "noise_magn", "noise_type", "filter","RMSE","MAPE", "R2", "F1", "REC", "PRE", "POR", "DATA", "ALG"])

    # Get the data name
    name_data = data_path.split('/')[-1].split('.')[0]

    # Load dataset
    dataset = pd.read_csv(data_path, sep=",", header=None)
    # Normalize columns for x and y data
    dataset = preprocessing(dataset, dataset.shape[1] - 1)
    # Get the data and targets into a container to generate the meshgrid
    dataset = Dataset(name_data, dataset[0], dataset[1])


    # Matrix of combined parameters for the experiments
    exp_params = np.array(np.meshgrid(  dataset,
                                        algorithms, 
                                        RandomForestRegressor(max_depth=9, random_state=0), 
                                        noise_params[0], # Number of added noisy instances (%)
                                        noise_params[1], # Percentage of noise added over the instance
                                        noise_params[2], # Noise type 
                                        noise_params[3]  # Filter 1 means yes
                                        )).T.reshape(-1, 7)

    experiment_on_noise(dataset, algorithms[1], RandomForestRegressor(max_depth=9, random_state=0), noise_params[0][0])
    results_df.head()

    # Reset the index in case addigng the rows caused indexing errors
    return results_df.reset_index(drop=True) 