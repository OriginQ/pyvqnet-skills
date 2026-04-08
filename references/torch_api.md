# PyTorch Backend API Reference

Starting from VQNet 2.15.0, VQNet supports using PyTorch as the backend for computation. This allows seamless integration with PyTorch models, existing code, and third-party libraries.

## Important Notes

After calling `pyvqnet.backends.set_backend("torch")`:
- `QTensor.data` changes from `pyvqnet._core.Tensor` to `torch.Tensor`
- You can use `to_tensor()` to wrap a `torch.Tensor` into `QTensor`
- When using PyTorch backend, all neural network modules must inherit from `pyvqnet.nn.torch.TorchModule` instead of the default
- Automatic differentiation variational quantum modules must inherit from `pyvqnet.qnn.vqc.torch.QModule`
- Calling `set_backend()` changes the **global** backend - QTensor created under different backends cannot interact with each other

## Backend Setting

### set_backend

```python
pyvqnet.backends.set_backend(backend_name)
```

**Description:**
Switches the global computation and storage backend. Allows choosing between:
- `pyvqnet` - PyVQNet C++ core, Python autograd
- `pyvqnet-ad` - PyVQNet C++ core, C++ autograd (default)
- `torch` - PyTorch core, PyTorch autograd - QTensor wraps torch.Tensor, API remains the same
- `torch-native` - Direct PyTorch output - accepts torch.Tensor or QTensor, returns torch.Tensor directly without conversion to QTensor

**Parameters:**
- `backend_name` - `str` - Backend name: "pyvqnet", "pyvqnet-ad", "torch", or "torch-native"

**Warning:** This changes the global backend. QTensor created under different backends cannot be used together in operations.

**Example:**
```python
import pyvqnet
pyvqnet.backends.set_backend("torch")
```

### get_backend

```python
pyvqnet.backends.get_backend(t=None)
```

**Description:**
Gets the current computation backend. If a QTensor is provided, returns the backend that was used to create that QTensor based on its `data` attribute.

**Parameters:**
- `t` - `QTensor`, optional - Tensor to query backend from, default: `None`
- **Returns:** `str` - Backend name. Default is "pyvqnet-ad". If t is provided, returns the backend for that tensor.

**Example:**
```python
import pyvqnet
pyvqnet.backends.set_backend("torch")
current = pyvqnet.backends.get_backend()
print(current)  # "torch"
```

## Backend Types Summary

| Backend | Description | Use Case |
|---------|-------------|----------|
| `pyvqnet-ad` | C++ core, C++ autograd | Default, best performance for most cases |
| `pyvqnet` | C++ core, Python autograd | Debugging |
| `torch` | PyTorch core, PyTorch autograd | Integration with PyTorch models |
| `torch-native` | Direct PyTorch, no QTensor wrapping | Maximum compatibility with PyTorch ecosystem |

## Example: PyTorch Backend Usage

```python
import pyvqnet
import torch
from pyvqnet.tensor import QTensor

# Switch to PyTorch backend
pyvqnet.backends.set_backend("torch")

# Create QTensor - now data is torch.Tensor
t = QTensor([1.0, 2.0, 3.0], requires_grad=True)
print(type(t.data))  # <class 'torch.Tensor'>

# Computations work the same as before
result = t.sum()
result.backward()
print(t.grad)
# QTensor still wraps everything - underlying computation is PyTorch
```

## Requirements

- PyTorch version **2.4.0 ~ 2.6.0** is required
- If using GPU PyTorch, CUDA 11.8 compatible version is required
- VQNet does **not** install PyTorch automatically - users must install it themselves
