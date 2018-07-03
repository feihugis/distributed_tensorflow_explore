import socket
import itertools
import operator

def get_hostname():
    return socket.gethostname()

def get_train_X_Y(iterator):
    X, Y = zip(*iterator)
    return X , Y

