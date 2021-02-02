import sys

import numpy as np
from functions import FUNCTIONS
from gwo import GWO


def optim(objf, wolf_count, iters, plot):
    print(f'optimizing {objf["name"]}:')
    best_result, convergence = GWO(objf, wolf_count, iters)
    print(f'   range: <{objf["low"]},{objf["high"]}>')
    print(f'   dimension: {objf["dim"]}')
    print(f'   optimum: {objf["opt"]}')
    print(f'   wolf count: {wolf_count}; iters: {iters}')
    print(
        f'   result: {best_result} (distance: {np.abs(objf["opt"] - best_result)})')

    if plot:
        import gnuplotlib as gp
        print('   convergence plot:')
        gp.plot(convergence, terminal='dumb 80,40',
                unset='grid', _with='lines')


if __name__ == '__main__':
    plot = not (len(sys.argv) > 1 and sys.argv[1] == '--no-gnuplot')

    for f in FUNCTIONS:
        optim(f(), 30, 500, plot)
