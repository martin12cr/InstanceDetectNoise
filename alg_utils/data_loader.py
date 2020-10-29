import pandas as pd

from sklearn.preprocessing import StandardScaler

"""
    This is function that loads the abalone dataset into a pandas dataframe,
    it also divides the dataframe into numpy arrays, x for the atributes and 
    y for the regression values

"""
def read_abalone(filename):

    # Read the column names
    df_names = pd.read_csv(filename, header=None, nrows=2, skiprows=10, sep=' ')
    # Add x values names
    col_names = df_names.iloc[0, 1:].tolist()
    # Add output row names
    col_names.append(df_names.iloc[1, 1])

    # Read the values and add the column names
    df = pd.read_csv(filename, header=None, skiprows=13, names=col_names)

    return df, df.iloc[:, :-1].values, df.iloc[:, -1].values.reshape(-1,1)


def preprocess_data(train_x, train_y, test_x, test_y):

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
    
    # Ravel the targets to make them suitable for training
    train_y = train_y.ravel()
    test_y = test_y.ravel()

    # Return processed data 
    return train_x, train_y, test_x, test_y
