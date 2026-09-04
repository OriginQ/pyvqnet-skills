# Utility Functions API Reference (v2.18.1)

## Random Seed Management

### set_random_seed

```python
pyvqnet.utils.set_random_seed(seed)
```

**Description:**
Sets the global random seed. Affects:
- `tensor.randu` - random uniform
- `tensor.randn` - random normal
- Parameter initialization for classical neural networks
- Parameter initialization for quantum computation layers

**Parameters:**
- `seed` - `int` - Random seed value

**Example:**
```python
import pyvqnet.tensor as tensor
from pyvqnet.utils import get_random_seed, set_random_seed

set_random_seed(256)

rn = tensor.randn([2, 3])
print(rn)
rn = tensor.randn([2, 3])
print(rn)
rn = tensor.randu([2, 3])
print(rn)
```

### get_random_seed

```python
pyvqnet.utils.get_random_seed()
```

**Description:**
Gets the current global random seed.

**Returns:** `int` - Current random seed

## Parameter Initializers

Available initializers in `pyvqnet.utils.initializer`:

- `he_normal` - He initialization (normal distribution)
- `he_uniform` - He initialization (uniform distribution)
- `xavier_normal` - Xavier/Glorot initialization (normal)
- `xavier_uniform` - Xavier/Glorot initialization (uniform)
- `uniform` - Uniform distribution
- `quantum_uniform` - Uniform initialization for quantum parameters
- `normal` - Normal distribution (Gaussian)

**Example:**
```python
from pyvqnet.nn.parameter import Parameter
from pyvqnet.utils.initializer import (
    he_normal, he_uniform,
    xavier_normal, xavier_uniform,
    uniform, quantum_uniform, normal
)

print(Parameter(shape=[2, 3], initializer=he_normal))
print(Parameter(shape=[2, 3], initializer=normal))
```
