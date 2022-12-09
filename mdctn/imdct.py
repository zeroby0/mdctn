import numpy as np
from mdctn import core

def imdct(signal, N=16, dct_type=4, norm='ortho', orthogonalize=None):
    z = np.asarray(signal)

    x = np.zeros(z.shape[0] + N//2)

    x[0:N//2] = core.imdct(z[-N//2:])[-N//2:]

    for i in np.r_[:z.shape[0]:N//2]:
        x[i:i+N] += core.imdct(z[i:i+N//2])

    return np.hstack([x[N//4:-N//2], x[:N//4]])
