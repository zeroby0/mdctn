import numpy as np
from scipy.fft import idct

__all__ = ['imdct']

def imdct(y, dct_type=4, norm='ortho', orthogonalize=True):
    """
    Returns Inverse Modified Discrete Cosine Transform of a 1 dimensional signal

    Parameters
    ----------
    y : array_like
        The input array
    dct_type : {1, 2, 3, 4}, optional
        Type of the DCT. Default is 4.
    norm : {'backward', 'ortho', 'forward'}, optional
        Normalisation mode for DCT. Default is 'ortho'
    orthogonalize: bool, optional
        Whether to use the orthogonalized DCT variant
        Defaults to ``True`` when ``norm=="ortho"`` and ``False`` otherwise.
    
    Returns
    -------
    z: ndarray of real
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
    N = y.shape[0] * 2

    if N%4 != 0:
        raise ValueError("Length of the input vector should be a multiple of 2.")
    
    N4 = N // 4

    z = idct(y, type=dct_type, norm=norm, orthogonalize=orthogonalize)

    z = np.hstack([z, -np.flip(z), -z])

    return z[N4:5*N//4]