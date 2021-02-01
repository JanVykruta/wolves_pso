import numpy as np


def F1():
    def f(x):
        return np.sum(x ** 2)

    return {'dim': 30, 'low': -100, 'high': 100, 'opt': 0, 'fnc': f, 'name': 'F1'}


def F2():
    def f(x):
        return np.sum(np.abs(x)) + np.prod(np.abs(x))

    return {'dim': 30, 'low': -10, 'high': 10, 'opt': 0, 'fnc': f, 'name': 'F2'}


def F7():
    dim = 30

    def f(x):
        return np.sum([i+1 for i in range(dim)] * (x ** 4))\
            + np.random.uniform(0, 1)

    return {'dim': dim, 'low': -1.28, 'high': 1.28, 'opt': 0, 'fnc': f, 'name': 'F7'}


def F11():
    dim = 30

    def f(x):
        w = [i + 1 for i in range(dim)]
        return np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(w))) + 1

    return {'dim': dim, 'low': -600, 'high': 600, 'opt': 0, 'fnc': f, 'name': 'F11'}
