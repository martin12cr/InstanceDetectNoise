from IS4BD.regENN import regENN
from alg_utils import data_loader as dl
from alg_utils import regression_models as rm




from sklearn.preprocessing import StandardScaler


def execute_regENN(train_files, test_files):

    # Read the training data 
    train_df, train_x, train_y = dl.read_abalone(train_files)
    # Read the test data
    test_df, test_x, test_y = dl.read_abalone(test_files)

    # Create a scaler for the data
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    # Fit on training data 
    x_scaler.fit(train_x)
    y_scaler.fit(train_y)
    # Scale and standarize
    train_x = x_scaler.transform(train_x)
    test_x = x_scaler.transform(test_x)
    train_y = y_scaler.transform(train_y)
    test_y = y_scaler.transform(test_y)

    print('Max y: ', max(test_y), '\tMin y: ', min(test_y))
    
    # Ravel the targets to make them suitable for training
    train_y = train_y.ravel()
    test_y = test_y.ravel()

    # Train a regressor 
    regressor = rm.train_regresor(train_x, train_y)

    # Execute the algorithm
    regENN(test_x, test_y, 1.5, 5, regressor)



training_files = 'Dataset/abalone/abalone-5-1tra.dat'
test_files = 'Dataset/abalone/abalone-5-1tst.dat'


execute_regENN(training_files, test_files)