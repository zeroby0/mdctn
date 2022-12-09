# MDCTN :yarn:

Multidimensional [Modified Discrete Cosine Transforms](https://en.wikipedia.org/wiki/Modified_discrete_cosine_transform)

```bash
pip install mdctn
```

- [x] 1-D MDCT & IMDCT
- [ ] n-D MDCT & IMDCT
- [ ] Windowing support
- [x] Helper functions for signals

Known bugs: Not all window sizes work: 12, 24, 28 don't work. Bug in arithmetic.

### 1-D MDCT on signals

Signals are wrapped around so all the data is stored in the same number of bits.

``` python
import numpy as np
from mdctn import mdct, imdct

x = np.arange(24)

y =  mdct(x, N=16)
z = imdct(y, N=16)

np.allclose(x, z) # True
```

### 1-D Pure MDCT

The core MDCT function

``` python
import numpy as np
from mdctn import core

x = np.arange(6) # [0, 1, 2, 3, 4, 5]

y_1 = core.mdct(x[0:4]) # [-2.50104055, -0.49476881]
y_2 = core.mdct(x[2:6]) # [-4.34879961, -1.26013568]

z_1 = core.imdct(y_1) # [-0.5,  0.5,  2.5,  2.5]
z_2 = core.imdct(y_2) #             [-0.5,  0.5,  4.5,  4.5]

z = (z_1[2:4] + z_2[0:2]) # [2.0, 3.0]
```

### Benchmarks

See [benchmarks.ipynb](./benchmarks.ipynb)




