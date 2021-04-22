import sys, time

import numpy as np
import pandas as pd
from sklearn.utils import Parallel

from tqdm import tqdm
from .istarmap import *
from multiprocessing import Pool

from .entities import Dataset, FoldIndex
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



def analyze_fold(dataset, fold_index, is_algorithm, aux_algoritm, noise_perc, noise_magn, noise_type, filtr):

    # Split train-test
    X_train, X_test = dataset.x[fold_index.train], dataset.x[fold_index.test]
    Y_train, Y_test = dataset.y[fold_index.train], dataset.y[fold_index.test]

    # Add noise to target
    Y_train, idNoise = add_noise(target=Y_train, perc=noise_perc, method=noise_type, mag=noise_magn)    


    if(filtr == 1):

        #print(Y_train.shape, Y_train.ravel().shape)
        
        # Apply instance selection algorithm
        idNoisePred = is_algorithm.evaluate(X_train,Y_train.ravel())

        # Delete Noise according to instance selection algorithmn
        X_train = X_train[idNoisePred != 1]
        Y_train = Y_train[idNoisePred != 1]

        # Evaluation Noise detection
        PRE = precision_score(idNoise, idNoisePred, pos_label=1, zero_division=0)
        REC = recall_score(idNoise, idNoisePred, pos_label=1, zero_division=0)
        F1 = f1_score(idNoise, idNoisePred, pos_label=1, zero_division=0)
        NUM = np.sum(idNoisePred)
    
    # Train the auxiliar algorithm
    aux_algoritm = aux_algoritm.fit(X_train, Y_train)

    # Generate prediction
    pred = aux_algoritm.predict(X_test)

    return [PRE, REC, NUM, F1, pred, Y_test]



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
def experiment_on_noise(dataset, is_algorithm, aux_algoritm, noise_perc, noise_magn=0.3, noise_type='persize', filtr=1, split=5,  rs=30, isParallel=False): 

    # KFold Cross Validation approach
    kf = KFold(n_splits=split, shuffle=True, random_state=rs)

    # Empty vectors to save predictions and real in each fold
    PREAGG=[]
    RECAGG=[]
    F1AGG=[]
    NUMAGG=[]
    predAgg= []
    YAgg= []

    folds = kf.split(dataset.x)

    # Check if we want to run on parallel mode
    if(isParallel):

        # Extract the folds for parallel processing 
        fold_indexes = []
        for train_index, test_index in folds:
            
            fold_indexes.append(FoldIndex(train_index, test_index))

        # Create a custom meshgrid for the parallel fold analysis
        exp_params = np.array(np.meshgrid(dataset,
                                        fold_indexes, 
                                        is_algorithm, 
                                        aux_algoritm, 
                                        noise_perc, 
                                        noise_magn, 
                                        noise_type, 
                                        filtr
                                        )).T.reshape(-1, 8)

        # Analyze each fold in parallel
        with Pool(split) as pool:
            results = list(pool.istarmap(analyze_fold, exp_params))

        # Append results
        for fold in results:

            PREAGG = np.append(PREAGG, fold[0])
            RECAGG = np.append(RECAGG, fold[1])
            NUMAGG = np.append(NUMAGG, fold[2])
            F1AGG = np.append(F1AGG, fold[3])
            predAgg = np.concatenate((predAgg, fold[4]), axis=0)
            YAgg = np.concatenate((YAgg, fold[5]), axis=0)

    # For secuential mode
    else:

        # Iterate over each train-test split
        for train_index, test_index in folds:

            
            # Run the fold analysis 
            results = analyze_fold(dataset, FoldIndex(train_index, test_index), is_algorithm, aux_algoritm, noise_perc, noise_magn, noise_type, filtr)
            

            # Store iteration results
            PREAGG = np.append(PREAGG, results[0])
            RECAGG = np.append(RECAGG, results[1])
            NUMAGG = np.append(NUMAGG, results[2])
            F1AGG = np.append(F1AGG, results[3])

            # Save real and prediction vectors of the  fold 
            predAgg = np.concatenate((predAgg, results[4]), axis=0)
            YAgg = np.concatenate((YAgg, results[5]), axis=0)

    
    # Evaluation
    RMSE = np.sqrt(mean_squared_error(YAgg, predAgg))
    MAPE = np.mean(np.abs(YAgg - predAgg) / (YAgg + 0.001))
    correlation_matrix = np.corrcoef(YAgg, predAgg)
    correlation_xy = correlation_matrix[0, 1]
    R2 = correlation_xy ** 2
    POR = 1 - (dataset.x[train_index].shape[0] / dataset.x.shape[0])
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
def experiment_on_data(algorithms, data_path, noise_params, isParallel=False):

    # Results container 
    results_df = pd.DataFrame(columns=["noise_perc", "noise_magn", "noise_type", "filter","RMSE","MAPE", "R2", "F1", "REC", "PRE", "POR", "DATA", "ALG"])

    # Get the data name
    name_data = data_path.split('/')[-1].split('.')[0]

    # Load dataset
    dataset = pd.read_csv(data_path, sep=",", header=None)
    #print(dataset.info())
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
    start = time.time()
    # If we wan multithreaded execution
    if(isParallel):
        # Start with the number of processors
        with Pool() as pool:
            # Run experiments in parallel and store results
            for noisy_result in  tqdm(pool.istarmap(experiment_on_noise, exp_params),
                                    total=len(exp_params), desc="Experimenting on " + name_data):

                # Add the results to the dataframe
                results_df.loc[len(results_df)] = noisy_result

    # If we want secuential execution
    else:

        # Add the parallelization on experiment_on_noise

        with tqdm(total=len(exp_params), desc="Experimenting on " + name_data) as pbar:
            for i in range(len(exp_params)):
                
                p = exp_params[i]
                # Add the results to the dataframe but excecute parallel folds
                results_df.loc[len(results_df)] = experiment_on_noise(p[0], p[1], p[2], p[3], p[4], p[5], p[6], isParallel=not(isParallel))

                pbar.update(1)
    
    print(time.time() - start)

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