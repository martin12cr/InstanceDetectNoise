
class Dataset:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

class FoldIndex:
    def __init__(self, train, test):
        self.train = train
        self.test = test

class Algorithm:
    def __init__(self, name, algorithm, **kwargs):
        # Store the function name
        self.name = name
        # Store the algorithm we want to call later
        self.algorithm = algorithm
        # Store kwargs for the algorithm parametrization
        self.kwargs = kwargs

    # Evaluate the algorithm using the kwargs
    def evaluate(self, x, y):

        return self.algorithm(x, y, **self.kwargs)
