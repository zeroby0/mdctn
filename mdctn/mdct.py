import numpy as np
from scipy.fft import dct

def mdct(x, N=16, **kwargs):
    """
    Returns Modified Discrete Cosine Transform of a 1 dimensional signal

    Parameters
    ----------
    x : array_like
        The input array
    N : integer, optional
        Window length. Skip length is half this.
        Default is 16.
    type : {1, 2, 3, 4}, optional
        Type of the DCT. Default is 4.
    norm : {'backward', 'ortho', 'forward'}, optional
        Normalisation mode for DCT. Default is 'ortho'
    orthogonalize: bool, optional
        Whether to use the orthogonalized DCT variant
        Defaults to ``True`` when ``norm=="ortho"`` and ``False`` otherwise.
        New since SciPy version 1.8.0
    
    Returns
    -------
    y: ndarray of real
        The transformed input array
    
    See Also
    --------
    imdct : Inverse Modified DCT

    Notes
    -----
    For details about normalisation modes of DCT, see scipy.fft.dct
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html#scipy.fft.dct

    For information about MDCT, see wikipedia / MDCT
    https://en.wikipedia.org/wiki/Modified_discrete_cosine_transform

    For understanding how MDCT is implemented, see appletonaudio.com
    https://www.appletonaudio.com/blog/2013/understanding-the-modified-discrete-cosine-transform-mdct/

    Examples
    --------
    >>> import numpy as np
    >>> from mdctn import mdct, imdct

    >>> x = np.arange(6) # [0, 1, 2, 3, 4, 5]
    >>> y = mdct(x[0:4]) # [-2.50104055, -0.49476881]
    >>> z = imdct(y) # [-0.5,  0.5,  2.5,  2.5]
    """
    npx = np.asarray(x)

    if np.issubdtype(npx.dtype, np.integer):
        npx = npx.astype(np.int64)
    elif np.issubdtype(npx.dtype, np.floating):
        npx = npx.astype(np.float64)
    else:
        raise ValueError(f'Unsupported dtype {npx.dtype} for mdct input. Please use np.int64 or np.float64.')

    N2 = N//2
    N4 = N//4

    if npx.shape[0] % N2 != 0:
        raise ValueError("Input array's length is not a multiple of half window length.")
    
    npx = np.hstack([npx, npx[:N2]])
    npx = npx.reshape(npx.shape[0]//N2, N2)
    npx = np.hstack([npx[:-1, :], npx[1:,:]])

    a = npx[:, 0*N4:1*N4]
    b = npx[:, 1*N4:2*N4]
    c = npx[:, 2*N4:3*N4]
    d = npx[:, 3*N4:4*N4]

    br = np.fliplr(b)
    cr = np.fliplr(c)

    npx = np.hstack([-cr - d, a - br])

    return dct(npx, **kwargs).ravel() / 2