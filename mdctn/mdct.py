import numpy as np
from scipy.fft import dct

__all__ = ['mdct']

def mdct(x, dct_type=4, norm='ortho', orthogonalize=None):
    """
    Returns Modified Discrete Cosine Transform of a 1 dimensional signal

    Parameters
    ----------
    x : array_like
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

    if orthogonalize is None:
        orthogonalize = False
        if norm == 'ortho': orthogonalize = True
    
    x = np.asarray(x)

    N = x.shape[0]

    if N%4 != 0:
        raise ValueError("Length of the input vector should be a multiple of 4.")

    N4 = N // 4

    a = x[0*N4:1*N4]
    b = x[1*N4:2*N4]
    c = x[2*N4:3*N4]
    d = x[3*N4:4*N4]

    br = np.flip(b)
    cr = np.flip(c)

    return dct(np.hstack([-cr - d, a - br]), type=dct_type, norm=norm, orthogonalize=orthogonalize) / 2