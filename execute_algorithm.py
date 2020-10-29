from algorithms.DROP2RE import DROP2RE
from alg_utils import data_loader as dl
from alg_utils import regression_models as rm



def execute_regENN(train_files, test_files):

    # Read the training data 
    train_df, train_x, train_y = dl.read_abalone(train_files)
    # Read the test data
    test_df, test_x, test_y = dl.read_abalone(test_files)

    # Preprocess data 
    train_x, train_y, test_x, test_y = dl.preprocess_data(train_x, train_y, test_x, test_y)

    # Train a regressor 
    #regressor = rm.train_regresor(train_x, train_y)

    # Execute the algorithm
    #regENN(test_x, test_y, 1.5, 5, regressor)
    DROP2RE(train_x, train_y, 11)



train_files = 'Dataset/abalone/abalone-5-1tra.dat'
test_files = 'Dataset/abalone/abalone-5-1tst.dat'


execute_regENN(train_files, test_files)


# TODO ALGORITHM BASED ON MUTUAL INFORMATION
# r^2