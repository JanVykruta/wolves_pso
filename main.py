import numpy as np
import gnuplotlib as gp
from functions import F1, F2, F7, F11
from gwo import GWO


def optim(objf, wolf_count, iters):
    print(f'optimizing {objf["name"]}:')
    best_result, convergence = GWO(objf, wolf_count, iters)
    print(f'   range: <{objf["low"]},{objf["high"]}>')
    print(f'   dimension: {objf["dim"]}')
    print(f'   optimum: {objf["opt"]}')
    print(f'   wolf count: {wolf_count}; iters: {iters}')
    print(
        f'   result: {best_result} (distance: {np.abs(objf["opt"] - best_result)})')
    print('   convergence plot:')
    gp.plot(convergence, terminal='dumb 80,40',
            unset='grid', _with='lines',
            _yrange=[-convergence[0] / 10.0, convergence[0] * 1.1])


if __name__ == '__main__':
    optim(F1(), 30, 500)
    optim(F2(), 30, 500)
    optim(F7(), 30, 500)
    optim(F11(), 30, 500)
