from datetime import datetime
from joblib import dump, load
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split



# Define the regressor parameters

# Random state used for reproductability in the data slit process
# and in the model weights generation
rs = 1

# SInce its a simple model use the lbfgs optimizer
optimizer ='sgd'

# Tolerance for early stopping
tolerance = 1e-5
max_no_chng = 10

# Activation function for the hidden layers
act = 'relu'

# Epochs
epochs = 100

# Learning rate and momentum
lr = 0.03
lr_pol = 'adaptive'
m = 0.02

def train_regresor(x, y, path='regresor_weights/abalone/test_'):

    # Divide the data into test and training
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=rs)


    #Number of hidden layers is (#attr+1)/2
    hl = int((x.shape[1] + 1) / 2)
    # Train the regressor
    model = MLPRegressor(hidden_layer_sizes=hl, activation=act, tol=tolerance, 
                        max_iter=epochs, learning_rate_init=lr, momentum=m,
                        learning_rate=lr_pol, n_iter_no_change=max_no_chng, 
                        random_state=rs, verbose=False, early_stopping=True).fit(x, y)

    preds = model.predict(x_test)
    print('Validation MAE: ', mean_absolute_error(y_test, preds))

    # Store the model
    now = datetime.now()
    #dump(model, path + now.strftime("%d-%m-%Y_%H:%M:%S") + '.joblib')

    return model

    
    

