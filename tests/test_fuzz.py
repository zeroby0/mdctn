import numpy as np
from itertools import product

from mdctn import mdct
from mdctn import imdct

def test_fuzz():
    for n in range(1, 20):
        N = 4 * n
        nsamples_s = np.arange(1,20) * N
        dct_types = [1, 2, 3, 4]
        norms = ['forward', 'backward', 'ortho']
        orthogonalizations = [True, False]

        for prod in product(nsamples_s, dct_types, norms, orthogonalizations):
            nsamples = prod[0]
            dct_type = prod[1]
            norm = prod[2]
            orthogonalize = prod[3]

            # print(nsamples, N, dct_type, norm, orthogonalize)

            x = np.random.randint(low=0, high=255, size=nsamples)

            y = mdct(x, N=N, type=dct_type, norm=norm, orthogonalize=orthogonalize)
            z = imdct(y, N=N, type=dct_type, norm=norm, orthogonalize=orthogonalize)

            assert np.allclose(x, z)