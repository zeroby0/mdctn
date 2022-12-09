import numpy as np
from mdctn import core

def mdct(signal, N=16, dct_type=4, norm='ortho', orthogonalize=None):
    np_signal = np.asarray(signal)

    # Wrap the signal around, for perfect reconstruction of the entire
    # signal without producing extra samples
    x = np.hstack([np_signal[-N//4:], np_signal, np_signal[:N//4]])

    transformed = np.zeros(np_signal.shape)

    for i in np.r_[:x.shape[0] - N//2:N//2]:
        transformed[i:i+N//2] = core.mdct(x[i:i+N])

    return transformed