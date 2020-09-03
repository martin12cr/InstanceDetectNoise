import pandas as pd



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
