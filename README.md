# MDCTN :yarn:

Multidimensional [Modified Discrete Cosine Transforms](https://en.wikipedia.org/wiki/Modified_discrete_cosine_transform)

```bash
pip install mdctn
```

- [x] 1-D MDCT & IMDCT
- [ ] n-D MDCT & IMDCT
- [ ] Windowing support
- [ ] Helper functions for signals

### 1-D MDCT

``` python
import numpy as np
from mdctn import mdct, imdct

x = np.arange(6) # [0, 1, 2, 3, 4, 5]

y_1 = mdct(x[0:4]) # [-2.50104055, -0.49476881]
y_2 = mdct(x[2:6]) # [-4.34879961, -1.26013568]

z_1 = imdct(y_1) # [-0.5,  0.5,  2.5,  2.5]
z_2 = imdct(y_2) #             [-0.5,  0.5,  4.5,  4.5]

z = (z_1[2:4] + z_2[0:2]) # [2.0, 3.0]
```

### Benchmarks

See [benchmarks.ipynb](./benchmarks.ipynb)




