# QTensor API Reference

QTensor is the core tensor data structure in VQNet with dynamic computation graph and automatic differentiation.

## Class Definition

```python
pyvqnet.tensor.tensor.QTensor(
    data,
    requires_grad=False,
    nodes=None,
    device=pyvqnet.DEV_CPU,
    dtype=None,
    name=""
)
```

**Parameters:**
- `data` - Input data, can be `_core.Tensor` or numpy array
- `requires_grad` - `bool` - Whether to track gradients, default: `False`
- `nodes` - List of successors in computation graph, default: `None`
- `device` - Storage device, default: `pyvqnet.DEV_CPU` (CPU)
- `dtype` - Data type, default: `None` (uses `kfloat32`)
- `name` - QTensor name, default: `""`
- **Returns:** `QTensor` - Output tensor

**Example:**
```python
from pyvqnet.tensor import QTensor
from pyvqnet.dtype import *
import numpy as np

t1 = QTensor(np.ones([2, 3]))
t2 = QTensor([2, 3, 4j, 5])
t3 = QTensor([[[2, 3, 4, 5], [2, 3, 4, 5]]], dtype=kbool)
print(t1)
# [[1. 1. 1.]
#  [1. 1. 1.]]
```

## Attributes

### `ndim`
Returns the number of dimensions.

```python
t.ndim -> int
```

**Example:**
```python
a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
print(a.ndim)  # 1
```

### `shape`
Returns the tensor dimensions as a list.

```python
t.shape -> List[int]
```

**Example:**
```python
a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
print(a.shape)  # [4]
```

### `size`
Returns the number of elements.

```python
t.size -> int
```

**Example:**
```python
a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
print(a.size)  # 4
```

### `numel()`
Returns the number of elements (same as `size`).

```python
t.numel() -> int
```

### `dtype`
Returns the tensor data type.

Supported data types:

| dtype | Description |
|-------|-------------|
| `pyvqnet.kbool` | Boolean |
| `pyvqnet.kuint8` | 8-bit unsigned integer |
| `pyvqnet.kint8` | 8-bit signed integer |
| `pyvqnet.kint16` | 16-bit signed integer |
| `pyvqnet.kint32` | 32-bit signed integer |
| `pyvqnet.kint64` | 64-bit signed integer |
| `pyvqnet.kfloat32` | 32-bit float (IEEE 754) |
| `pyvqnet.kfloat64` | 64-bit double |
| `pyvqnet.kcomplex64` | 64-bit complex (two float32) |
| `pyvqnet.kcomplex128` | 128-bit complex (two float64) |
| `pyvqnet.kbfloat16` | 16-bit bfloat (brain floating point) |

### `is_contiguous`
Whether the tensor is contiguous in memory.

```python
t.is_contiguous -> bool
```

**Example:**
```python
a = QTensor([[2, 3, 4, 5], [2, 3, 4, 5]])
print(a.is_contiguous)  # True
c = a.permute((1, 0))
print(c.is_contiguous)  # False
```

### `requires_grad`
Whether gradients are tracked for this tensor.

### `grad`
Contains the gradient after `backward()` is called. Shape matches the tensor.

## Methods

### `zero_grad()`
Sets the gradient to zero. Used in optimization loops.

```python
t.zero_grad() -> None
```

**Example:**
```python
t3 = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
t3.zero_grad()
print(t3.grad)  # [0., 0., 0., 0.]
```

### `backward(grad=None)`
Computes gradients via backpropagation for all tensors in the computation graph that require gradients.

```python
t.backward(grad=None) -> None
```

**Example:**
```python
target = QTensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0.2]], requires_grad=True)
y = 2 * target + 3
y.backward()
print(target.grad)
# [[2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]]
```

### `to_numpy()`
Copies the tensor data into a `numpy.ndarray`.

```python
t.to_numpy() -> np.ndarray
```

### `to(device)`
Moves the tensor to the specified device (CPU/GPU).

### `contiguous()`
Makes the tensor contiguous in memory.

## Common Creation Functions

Most common factory functions:

- `pyvqnet.tensor.ones(shape)` - Tensor of all ones
- `pyvqnet.tensor.zeros(shape)` - Tensor of all zeros
- `pyvqnet.tensor.randn(shape)` - Random normal distribution
- `pyvqnet.tensor.randu(shape)` - Random uniform distribution
- `pyvqnet.tensor.full(shape, value)` - Fill with constant value
- `pyvqnet.tensor.eye(n)` - Identity matrix

## Data Types Example

```python
from pyvqnet.tensor import QTensor
from pyvqnet.dtype import *

a = QTensor([2, 3, 4, 5])
print(a.dtype)  # kfloat32 (default)
```
