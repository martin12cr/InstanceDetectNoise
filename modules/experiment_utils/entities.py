
class Dataset:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

class Algorithm:
    def __init__(self, name, evaluate):
        self.name = name
        self.evaluate = evaluate
