import numpy as np


def GWO(objf, wolf_count, iters):
    dim = objf['dim']
    alpha_pos = None
    beta_pos = None
    delta_pos = None

    alpha = float("inf")
    beta = alpha
    delta = beta

    lower_bound = objf['low']
    upper_bound = objf['high']

    wolves = np.random.uniform(
        lower_bound, upper_bound, size=(wolf_count, dim))
    convergence = []

    for iter in range(iters):
        for i in range(wolf_count):
            wolves[i, :] = np.clip(wolves[i, :], lower_bound, upper_bound)

            obj_val = objf['fnc'](wolves[i, :])

            if obj_val < alpha:
                alpha = obj_val
                alpha_pos = wolves[i, :]
            elif obj_val < beta:
                beta = obj_val
                beta_pos = wolves[i, :]
            elif obj_val < delta:
                delta = obj_val
                delta_pos = wolves[i, :]

        a = 2 - iter * (2 / iters)

        def get_X_for(pos):
            r1 = np.random.uniform(size=(wolf_count, dim))
            r2 = np.random.uniform(size=(wolf_count, dim))

            A = 2 * a * r1 - a
            C = 2 * r2

            D = np.abs(C * pos - wolves)

            return pos - A * D

        X1 = get_X_for(alpha_pos)
        X2 = get_X_for(beta_pos)
        X3 = get_X_for(delta_pos)

        wolves = (X1 + X2 + X3) / 3

        convergence.append(alpha)

    return alpha, np.array(convergence)
