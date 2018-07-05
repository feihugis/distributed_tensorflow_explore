import socket
import itertools
import operator
import numpy as np
def get_hostname():
    return socket.gethostname()

def get_train_X_Y(iterator):
    X, Y = zip(*iterator)
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X , Y

