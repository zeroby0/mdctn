import numpy as np
from scipy.fft import idct

def imdct(y, N=16, **kwargs):
    """
    Returns Inverse Modified Discrete Cosine Transform of a 1 dimensional signal

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
    mdct : Modified DCT

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
    npy = np.asarray(y)

    N2 = N//2
    N4 = N//4

    if npy.shape[0] % N2 != 0:
        raise ValueError("Input array's length is not a multiple of half window length.")
    
    npy = npy.reshape(npy.shape[0]//N2, N2)

    z = idct(npy, **kwargs)
    z = np.hstack([z[:, N4:], -np.fliplr(z), -z[:, :N4]])

    p = np.roll(z, shift=-1, axis=0)

    return np.roll(z[:, -N2:] + p[:, :N2], shift=1, axis=0).ravel()