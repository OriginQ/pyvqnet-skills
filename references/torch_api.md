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

---

## Torch Bridge NN Modules (`pyvqnet.nn.torch`)

After calling `pyvqnet.backends.set_backend("torch")`, the following modules from `pyvqnet.nn.torch` become available. Each module inherits from both `pyvqnet.nn.Module` and `torch.nn.Module`, enabling seamless integration with PyTorch training loops.

**Import path:** `from pyvqnet.nn.torch import ...`

### TorchModule — Base Class for Torch Backend

```python
pyvqnet.nn.torch.TorchModule(*args, **kwargs)
```

**Description:**
Base class for defining neural network modules when using PyTorch backend. Inherits from both `pyvqnet.nn.Module` and `torch.nn.Module`. Parameters in TorchModule are `torch.Tensor` type (or `torch.nn.Parameter`).

**Key features:**
- Parameters are stored as `torch.nn.Parameter` in `self._parameters`
- Buffers are stored as `torch.Tensor` in `self._buffers`
- Supports `.to(device)`, `.to_gpu()`, `.to_cpu()` for device transfer
- `.train()` / `.eval()` methods for training/evaluation mode
- `.register_parameter(name, param)` to register parameters manually

**Warning:** Requires backend to be `"torch"` or `"torch-native"`. Asserts at init time.

**Example:**
```python
import pyvqnet
pyvqnet.backends.set_backend("torch")
from pyvqnet.nn.torch import TorchModule, Linear

class MyModel(TorchModule):
    def __init__(self):
        super().__init__()
        self.fc = Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
```

### TorchSequential

```python
pyvqnet.nn.torch.TorchSequential(*args)
```

**Description:**
Sequential container for Torch backend modules. Accepts modules as positional arguments or an `OrderedDict` of named modules. Inherits from `TorchModule` and `Sequential`.

**Example:**
```python
from pyvqnet.nn.torch import TorchSequential, Conv2D, ReLu

model = TorchSequential(
    Conv2D(1, 20, (5, 5)),
    ReLu(),
    Conv2D(20, 64, (5, 5)),
    ReLu(),
)

# With OrderedDict
from collections import OrderedDict
model = TorchSequential(OrderedDict([
    ('conv1', Conv2D(1, 20, (5, 5))),
    ('relu1', ReLu()),
]))
```

### TorchModuleList

```python
pyvqnet.nn.torch.TorchModuleList(modules=None)
```

**Description:**
Holds submodules in a list for the Torch backend. Inherits from `ModuleList` and `torch.nn.ModuleList`. Modules can be accessed by index.

**Example:**
```python
from pyvqnet.nn.torch import TorchModule, Linear, TorchModuleList

class M(TorchModule):
    def __init__(self):
        super().__init__()
        self.layers = TorchModuleList([Linear(4, 1), Linear(4, 1)])

    def forward(self, x):
        return self.layers[0](x) + self.layers[1](x)
```

### TorchModuleDict

```python
pyvqnet.nn.torch.TorchModuleDict(modules=None)
```

**Description:**
Holds submodules in a dictionary for the Torch backend. Supports `__getitem__`, `__setitem__`, `__delitem__`, `.keys()`, `.values()`, `.items()`, `.update()`, `.pop()`, `.clear()`.

### TorchParameterList

```python
pyvqnet.nn.torch.TorchParameterList(values=None)
```

**Description:**
Holds parameters in a list for the Torch backend. Each element is a `Parameter` (QTensor wrapping torch.nn.Parameter).

**Example:**
```python
from pyvqnet.nn.torch import TorchModule, TorchParameterList
import pyvqnet.nn as nn

class MyModule(TorchModule):
    def __init__(self):
        super().__init__()
        self.params = TorchParameterList([nn.Parameter((10, 10)) for _ in range(10)])

    def forward(self, x):
        for i, p in enumerate(self.params):
            x = self.params[i // 2] * x + p * x
        return x
```

### TorchParameterDict

```python
pyvqnet.nn.torch.TorchParameterDict(values=None)
```

**Description:**
Holds parameters in a dictionary for the Torch backend. Inherits from `TorchModule` and `ParameterDict`.

---

### Linear (TorchLinear)

```python
pyvqnet.nn.torch.Linear(
    input_channels: int,
    output_channels: int,
    weight_initializer: Callable = None,
    bias_initializer: Callable = None,
    use_bias: bool = True,
    dtype=None,
    name: str = "",
)
```

**Alias:** `TorchLinear = Linear`

**Description:**
Fully connected (dense) layer: `y = x @ A.T + b`. Inherits from `TorchModule`. Weight is `torch.nn.Parameter` of shape `(output_channels, input_channels)`, initialized with `he_uniform`. Bias initialized with `zeros`.

**Example:**
```python
import pyvqnet
from pyvqnet.tensor import QTensor
from pyvqnet.nn.torch import Linear
import numpy as np

pyvqnet.backends.set_backend("torch")
n = Linear(4, 2)
x = QTensor(np.arange(24).reshape(2, 3, 4), requires_grad=True, dtype=pyvqnet.kfloat32)
y = n(x)
```

---

### Conv1D

```python
pyvqnet.nn.torch.Conv1D(
    input_channels: int,
    output_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: Union[str, int] = "valid",
    use_bias: bool = True,
    kernel_initializer: Callable = None,
    bias_initializer: Callable = None,
    dilation_rate: int = 1,
    group: int = 1,
    dtype=None,
    name: str = "",
)
```

**Description:**
1D convolution layer for Torch backend. Supports `"valid"` / `"same"` padding strings or integer padding. Kernel shape: `(output_channels, input_channels // group, kernel_size)`.

### Conv2D / Conv2d

```python
pyvqnet.nn.torch.Conv2D(
    input_channels: int,
    output_channels: int,
    kernel_size: Union[Tuple[int, int], List[int], int],
    stride: Union[Tuple[int, int], List[int], int] = (1, 1),
    padding: Union[Tuple[int, int], List[int], int, str] = "valid",
    use_bias: bool = True,
    kernel_initializer: Callable = None,
    bias_initializer: Callable = None,
    dilation_rate: Union[Tuple[int, int], List[int], int] = (1, 1),
    group: int = 1,
    dtype=None,
    name: str = "",
)
```

**Alias:** `Conv2d = Conv2D`

**Description:**
2D convolution layer for Torch backend. Kernel shape: `(output_channels, input_channels // group, kernel_h, kernel_w)`.

### ConvT2D (Transposed Conv2D)

```python
pyvqnet.nn.torch.ConvT2D(
    input_channels: int,
    output_channels: int,
    kernel_size: Union[Tuple[int, int], List[int], int],
    stride: Union[Tuple[int, int], List[int], int] = (1, 1),
    padding: Union[Tuple[int, int], List[int], int] = (0, 0),
    use_bias: bool = True,
    kernel_initializer: Callable = None,
    bias_initializer: Callable = None,
    dilation_rate: Union[Tuple[int, int], List[int], int] = (1, 1),
    out_padding: Union[Tuple[int, int], List[int], int] = (0, 0),
    group: int = 1,
    dtype=None,
    name: str = "",
)
```

**Description:**
Transposed 2D convolution for Torch backend. Does **not** support string padding — use tuple of ints. Kernel shape: `(input_channels, output_channels // group, kernel_h, kernel_w)`.

---

### Pooling Layers

```python
pyvqnet.nn.torch.MaxPool1D(kernel, stride, padding=0, name="")
pyvqnet.nn.torch.MaxPool2D(kernel, stride, padding=(0,0), name="")
pyvqnet.nn.torch.AvgPool1D(kernel, stride, padding=0, name="")
pyvqnet.nn.torch.AvgPool2D(kernel, stride, padding=(0,0), name="")
pyvqnet.nn.torch.AdaptiveAvgPool2d(out_size, name="")
```

**Description:**
Standard pooling layers wrapping `torch.nn.functional`. All inherit from `TorchModule`.

---

### Normalization Layers

```python
pyvqnet.nn.torch.BatchNorm1d(channel_num, momentum=0.1, epsilon=1e-5, affine=True, ...)
pyvqnet.nn.torch.BatchNorm2d(channel_num, momentum=0.1, epsilon=1e-5, affine=True, ...)
pyvqnet.nn.torch.LayerNorm(normalized_shape, epsilon=1e-5, affine=True, ...)
pyvqnet.nn.torch.LayerNorm1d(normalized_shape, epsilon=1e-5, affine=True, ...)
pyvqnet.nn.torch.LayerNorm2d(normalized_shape, epsilon=1e-5, affine=True, ...)
pyvqnet.nn.torch.LayerNormNd(normalized_shape, epsilon=1e-5, affine=True, ...)
pyvqnet.nn.torch.GroupNorm(num_groups, num_channels, epsilon=1e-5, affine=True, ...)
```

**Aliases:**
- `BatchNorm1d = BatchNormNd`, `BatchNorm2d = BatchNormNd`
- `LayerNorm = LayerNormNd`

**Description:**
Normalization layers for Torch backend. All inherit from `TorchModule` and wrap `torch.nn.functional` calls.

- **BatchNormNd:** Applies Batch Normalization over a mini-batch. Tracks running mean/variance during training. Supports cumulative and exponential moving average.
- **LayerNormNd:** Applies Layer Normalization over the last D dimensions. `LayerNorm2d` asserts 4D input, `LayerNorm1d` asserts 2D input.
- **GroupNorm:** Applies Group Normalization. Input channels are separated into `num_groups` groups.

---

### Dropout Layers

```python
pyvqnet.nn.torch.Dropout(dropout_rate=0.5, name="")
pyvqnet.nn.torch.DropPath(dropout_rate=0.5, name="")
```

**Description:**
- **Dropout:** Randomly zeroes elements with probability `dropout_rate`. Wraps `torch.nn.functional.dropout`.
- **DropPath:** Stochastic Depth (DropPath) module, randomly drops residual branches.

---

### Activation Functions

```python
pyvqnet.nn.torch.ReLu(name="")           # ReLU (Alias: ReLU)
pyvqnet.nn.torch.LeakyReLu(alpha=0.01)   # LeakyReLU
pyvqnet.nn.torch.ELU(alpha=1.0)          # ELU
pyvqnet.nn.torch.Tanh(name="")           # Tanh
pyvqnet.nn.torch.Sigmoid(name="")        # Sigmoid
pyvqnet.nn.torch.Softmax(dim=-1)         # Softmax
pyvqnet.nn.torch.Softplus(name="")       # Softplus (beta=1, threshold=2000000)
pyvqnet.nn.torch.Softsign(name="")       # Softsign
pyvqnet.nn.torch.HardSigmoid(name="")    # HardSigmoid
pyvqnet.nn.torch.Gelu(approximate="tanh")# GELU (Alias: GeLU)
pyvqnet.nn.torch.SiLU(name="")           # SiLU (Swish)
pyvqnet.nn.torch.SwiGLU(name="")         # SwiGLU (takes gate, up inputs)
```

**Description:**
Common activation functions as TorchModule subclasses. All wrap `torch.nn.functional` operations.

- **Gelu** supports `approximate` parameter: `"tanh"` (default) or `"none"`
- **Softplus** uses `beta=1, threshold=2000000`
- **SwiGLU** takes two inputs `(gate, up)` and computes `silu(gate) * up`
- **GeLU** is an alias for `Gelu`

---

### Embedding

```python
pyvqnet.nn.torch.Embedding(
    num_embeddings: int,
    embedding_dim: int,
    weight_initializer: Callable = xavier_normal,
    dtype=None,
    name: str = "",
)
```

**Description:**
Word embedding layer. Stores embeddings in a `(num_embeddings, embedding_dim)` weight matrix. Input is a list of indices (requires `kint64` dtype). Wraps `torch.nn.functional.embedding`.

---

### RNN / LSTM / GRU

```python
pyvqnet.nn.torch.RNN(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    nonlinearity: str = "tanh",     # "tanh" or "relu"
    batch_first: bool = True,
    use_bias: bool = True,
    bidirectional: bool = False,
    dtype=None,
    name: str = "",
)

pyvqnet.nn.torch.LSTM(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    batch_first: bool = True,
    use_bias: bool = True,
    bidirectional: bool = False,
    dtype=None,
    name: str = "",
)

pyvqnet.nn.torch.GRU(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    batch_first: bool = True,
    use_bias: bool = True,
    bidirectional: bool = False,
    dtype=None,
    name: str = "",
)
```

**Description:**
Recurrent neural network layers for fixed-length sequences. Each inherits from `TorchModule` and the corresponding VQNet NN module.

**Forward signature:**
```python
output, hidden_state = rnn(x, init_states)
```

- **RNN:** `h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)` (or ReLU)
- **LSTM:** Returns `(output, (h_n, c_n))` — hidden state and cell state
- **GRU:** Returns `(output, h_n)` — output and hidden state

**Example:**
```python
from pyvqnet.nn.torch import GRU
from pyvqnet.tensor import tensor

rnn = GRU(4, 6, 2, batch_first=False, bidirectional=True)
x = tensor.ones([5, 3, 4])   # (seq_len, batch, input_size)
h0 = tensor.ones([4, 3, 6])  # (num_layers * num_directions, batch, hidden_size)
output, hn = rnn(x, h0)
```

**Dynamic variants** (for variable-length sequences using `PackedSequence`):

```python
pyvqnet.nn.torch.Dynamic_RNN(input_size, hidden_size, ...)
pyvqnet.nn.torch.Dynamic_LSTM(input_size, hidden_size, ...)
pyvqnet.nn.torch.Dynamic_GRU(input_size, hidden_size, ...)
```

These accept and return `PackedSequence` objects, constructed via `tensor.pad_sequence` + `tensor.pack_pad_sequence`.

---

### SDPA — Scaled Dot Product Attention

```python
pyvqnet.nn.torch.SDPA(
    attn_mask=None,
    dropout_p=0.0,
    scale=None,
    is_causal=False,
)
```

**Description:**
Scaled dot product attention layer. On CPU, uses a native implementation; on GPU, delegates to `torch.nn.functional.scaled_dot_product_attention`. Inherits from `TorchModule` and the VQNet transformer `SDPA`.

**Forward:**
```python
output = sdpa(query, key, value)
```

**Parameters:**
- `attn_mask` — Attention mask (broadcastable to attention weights shape)
- `dropout_p` — Dropout probability applied to attention weights
- `scale` — Scaling factor before softmax (default: `1 / sqrt(dim)`)
- `is_causal` — If True, applies causal masking (lower triangular). Cannot be used with `attn_mask`.

---

### SwinTransformer

```python
pyvqnet.nn.torch.SwinTransformer(
    patch_size: List[int],
    embed_dim: int,
    depths: List[int],
    num_heads: List[int],
    window_size: List[int],
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
    attention_dropout: float = 0.0,
    stochastic_depth_prob: float = 0.1,
    num_classes: int = 1000,
    norm_layer=None,
    block=None,
    downsample_layer=PatchMerging,
    feat_dim: int = 16,
)

pyvqnet.nn.torch.swin_b(weights=None, **kwargs)
```

**Description:**
Full Swin Transformer implementation from "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (CVPR 2021). Supports shifted window attention, patch merging, stochastic depth.

- **`swin_b()`** — Convenience function to create a Swin-Base model with:
  - `patch_size=[4, 4]`, `embed_dim=128`
  - Depths: `[2, 2, 18, 2]`, heads: `[4, 8, 16, 32]`
  - `window_size=[7, 7]`, `stochastic_depth_prob=0.5`

**Internal components (also exported):**
- `ShiftedWindowAttention` — Shifted window multi-head self-attention
- `SwinTransformerBlock` — Swin Transformer block with attention + MLP
- `PatchMerging` — Patch merging downsampling layer
- `StochasticDepth` — Stochastic depth regularization

---

### Pixel Shuffle / Unshuffle

```python
pyvqnet.nn.torch.Pixel_Shuffle(upscale_factors: int)
pyvqnet.nn.torch.Pixel_Unshuffle(downscale_factors: int)
```

**Description:**
Rearranges elements in a tensor of shape `(N, C * r^2, H, W)` to `(N, C, H * r, W * r)` (shuffle) or the inverse (unshuffle). Wraps `torch.nn.functional.pixel_shuffle` / `pixel_unshuffle`.

---

### Interpolate

```python
pyvqnet.nn.torch.Interpolate(
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
    name="",
)
```

**Description:**
Down/up-samples input to given size or scale factor. Modes: `"nearest"`, `"bilinear"`, `"bicubic"`. Supports 4D input `(N, C, H, W)`. Wraps `torch.nn.functional.interpolate`.

---

### Loss Functions

Loss functions in the Torch bridge follow the VQNet convention: `loss_fn(target, output)` (target/truth first).

```python
pyvqnet.nn.torch.MeanSquaredError(name="")
pyvqnet.nn.torch.CategoricalCrossEntropy(name="")
pyvqnet.nn.torch.SoftmaxCrossEntropy(name="")
pyvqnet.nn.torch.BinaryCrossEntropy(name="")
pyvqnet.nn.torch.NLL_Loss(name="")
pyvqnet.nn.torch.CrossEntropyLoss(name="")
```

**Aliases:** `TorchMeanSquaredError = MeanSquaredError`, `TorchCategoricalCrossEntropy = CategoricalCrossEntropy`, etc.

| Loss | Formula | Forward Args |
|------|---------|-------------|
| `MeanSquaredError` | `mean((x - y)^2)` | `(target, output)` |
| `CategoricalCrossEntropy` | `-log(exp(x[c]) / sum(exp(x)))` | `(target, output)` — argmax on target |
| `SoftmaxCrossEntropy` | Numerically stable LogSoftmax + NLLLoss | `(target, output)` |
| `BinaryCrossEntropy` | `-w * [y * log(x) + (1-y) * log(1-x)]` | `(target, output)` |
| `NLL_Loss` | `-sum(output[target]) / N` | `(target, output)` — expects log-probabilities |
| `CrossEntropyLoss` | LogSoftmax + NLLLoss | `(target, output)` — raw scores |

---

## Torch Quantum Operations (`pyvqnet.torch.quantum`)

```python
from pyvqnet.torch.quantum import *
```

These are low-level quantum operations used internally by the VQC autograd engine when the PyTorch backend is active.

### expval_pauli

```python
pyvqnet.torch.quantum.expval_pauli(q_machine, obs)
```

**Description:**
Compute the expectation value of a Pauli operator string. Automatically dispatches to CUDA-accelerated path if available (`_EXPVAL_HAMI_AVAILABLE`). Falls back to matmul-based path.

**Parameters:**
- `q_machine` — Quantum machine (QVM)
- `obs` — dict with `"wires"`, `"observables"`, `"coefficient"` keys

### ry_matrix / crz_matrix

```python
pyvqnet.torch.quantum.ry_matrix(params)
pyvqnet.torch.quantum.crz_matrix(params)
```

**Description:**
Compute unitary matrices for RY and CRZ gates using efficient batched tensor operations.

- **ry_matrix:** Returns `(batch, 2, 2)` complex unitary matrix for RY(θ)
- **crz_matrix:** Returns `(batch, 4, 4)` complex unitary matrix for CRZ(θ)

### marginal_prob / probs

```python
pyvqnet.torch.quantum.marginal_prob(prob, num_wires, wires)
pyvqnet.torch.quantum.probs(q_state, num_wires, wires)
```

**Description:**
- **probs:** Compute marginal probabilities from quantum state
- **marginal_prob:** Compute marginal probabilities by summing over inactive wires

### _apply_unitary_bmm

```python
pyvqnet.torch.quantum._apply_unitary_bmm(mat, state, permute_to, permute_back, original_shape, name)
```

**Description:**
Apply a unitary matrix to a quantum state using batched matrix multiplication (`bmm`). Supports broadcasting when matrix is not batched.

### C++ CUDA Kernel Operations

When the `extension_cpp_for_qllm` extension is available, these operations use optimized CUDA kernels:

```python
# Single-qubit parameterized gates
bs_1p_ry_forward(state, theta, obj_qubit, ctrl_qubits=None)
bs_1p_rx_forward(state, theta, obj_qubit, ctrl_qubits=None)
bs_1p_rz_forward(state, theta, obj_qubit, ctrl_qubits=None)

# Controlled parameterized gates
bs_1p_crz_forward(state, theta, obj_qubit, ctrl_qubits=None)
bs_1p_crx_forward(state, theta, obj_qubit, ctrl_qubits=None)
bs_1p_cry_forward(state, theta, obj_qubit, ctrl_qubits=None)

# Fused (batched) operations
bs_1p_fused_linear_crz_forward(state, thetas, obj_qubits, ctrls)
bs_1p_fused_crx_forward(state, thetas, obj_qubits, ctrls)
bs_1p_fused_cry_forward(state, thetas, obj_qubits, ctrls)

# CNOT
cnot_forward(state, obj_qubit, ctrl_qubits)
bs_fused_cnot_forward(state, obj_qubits, ctrls)
```

**Description:**
Lazy-loaded C++/CUDA extension for high-performance quantum gate simulation. All functions require contiguous inputs. The `bs_` prefix indicates batched-state operations.

---

## torch 后端状态向量接口 (`pyvqnet.qnn.vqc.sv.torch`)

在调用 `pyvqnet.backends.set_backend("torch")` 后，可以使用 `pyvqnet.qnn.vqc.sv.torch` 下的态矢(state vector)变分量子线路模块。编写变分量子线路模型需要继承于 `pyvqnet.qnn.vqc.sv.torch.QModule`，其中计算使用 `torch.Tensor` 进行，参数可通过 torch 自动微分训练。

**Import path:** `from pyvqnet.qnn.vqc.sv.torch import ...`

**Warning:** 这些类以及其派生类仅适用于 `pyvqnet.backends.set_backend("torch")`，不要与默认 `pyvqnet.nn` 下的 `Module` 混用。这些类的非参数成员变量 `_buffers` 中的数据为 `torch.Tensor` 类型；参数成员变量 `_parameters` 中的数据为 `torch.nn.Parameter` 类型。

### QModule

```python
pyvqnet.qnn.vqc.sv.torch.QModule(name="")
```

**Description:**
当用户使用 `torch` 后端时，定义量子变分线路模型 `Module` 应该继承的基类。该类继承于 `pyvqnet.nn.torch.TorchModule` 以及 `torch.nn.Module`，可以作为 `torch.nn.Module` 的一个子模块加入 torch 的模型中。

**Note:**
- 该类以及其派生类仅适用于 `pyvqnet.backends.set_backend("torch")`，不要与默认 `pyvqnet.nn` 下的 `Module` 混用
- 该类的 `_buffers` 中的数据为 `torch.Tensor` 类型；`_parameters` 中的数据为 `torch.nn.Parameter` 类型

### QMachine

```python
pyvqnet.qnn.vqc.sv.torch.QMachine(num_wires, dtype=pyvqnet.kcomplex64,grad_mode="",save_ir=False)
```

**Description:**
变分量子计算的模拟器类，包含 `states` 属性为量子线路的 statevectors。该类继承于 `pyvqnet.nn.torch.TorchModule` 以及 `pyvqnet.qnn.QMachine`，可以作为 `torch.nn.Module` 的一个子模块加入 torch 的模型中。

**Warning:** 在每次运行一个完整的量子线路之前，必须使用 `pyvqnet.qnn.vqc.sv.torch.QMachine.reset_states(batchsize)` 将模拟器里面初态重新初始化，并且广播为 `(batchsize,*)` 维度从而适应批量数据训练。

**Parameters:**
- `num_wires` - 量子比特数
- `dtype` - 计算数据的数据类型。默认值是 `pyvqnet.kcomplex64`，对应的参数精度为 `pyvqnet.kfloat32`
- `grad_mode` - 梯度计算模式，可为 `"adjoint"`，默认值 `""`，使用自动微分模拟
- `save_ir` - 设置为 True 时，将操作保存到 originIR，默认值 False

**Example:**
```python
from pyvqnet.qnn.vqc.sv.torch import QMachine
import pyvqnet
pyvqnet.backends.set_backend("torch")
qm = QMachine(4)
print(qm.states)
```

**Methods:**

```python
QMachine.reset_states(batchsize)
```

将模拟器里面初态重新初始化，并且广播为 `(batchsize,*)` 维度从而适应批量数据训练。

- `batchsize` - 批处理维度

### 变分量子逻辑门

以下 `pyvqnet.qnn.vqc.sv.torch` 中的量子线路模块继承于 `pyvqnet.qnn.vqc.sv.torch.QModule`，其中计算使用 `torch.Tensor` 进行计算。各逻辑门同时提供小写函数形式（如 `rx`、`ry`、`rz`、`cnot` 等，见上方 csv 支持列表 `same_apis_from_vqc.csv`），可直接作用于 `QMachine`。

所有逻辑门类共享如下签名（以 `I` 为例，其余仅类名不同）：

```python
pyvqnet.qnn.vqc.sv.torch.I(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.Hadamard(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.T(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.S(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.PauliX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.PauliY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.PauliZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.X1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.RX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.RY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.RZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.CRX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.CRY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.CRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.U1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.U2(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.U3(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.CNOT(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.CY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.CZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.CR(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.SWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.CSWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.RXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.RYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.RZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.RZX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.Toffoli(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.IsingXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.IsingYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.IsingZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.IsingXY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.PhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.MultiRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.SDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.TDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.ControlledPhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.sv.torch.MultiControlledX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False,control_values=None)
```

**Parameters (all gates):**
- `has_params` - 是否具有参数，例如 RX,RY 等门需要设置为 True，不含参数的需要设置为 False，默认为 False
- `trainable` - 是否自带含待训练参数，如果该层使用外部输入数据构建逻辑门矩阵，设置为 False，如果待训练参数需要从该层初始化，则为 True，默认为 False
- `init_params` - 初始化参数，用来编码经典数据 QTensor，默认为 None
- `wires` - 线路作用的比特索引，默认为 None
- `dtype` - 逻辑门内部矩阵的数据精度，可以设置为 `pyvqnet.kcomplex64`，或 `pyvqnet.kcomplex128`，分别对应 float 输入或者 double 入参
- `use_dagger` - 是否使用该门的转置共轭版本，默认为 False
- `control_values` - 仅 `MultiControlledX`：控制值，默认为 None，当比特位为 1 时控制

**Gate list and example instantiation (from official doc):**

| Gate | Description | Example instantiation |
|------|-------------|----------------------|
| `I` | 定义一个I逻辑门类 | `layer = I(wires=0)` |
| `Hadamard` | 定义一个Hadamard逻辑门类 | `layer = Hadamard(wires=0)` |
| `T` | 定义一个T逻辑门类 | `layer = T(wires=0)` |
| `S` | 定义一个S逻辑门类 | `layer = S(wires=0)` |
| `PauliX` | 定义一个PauliX逻辑门类 | `layer = PauliX(wires=0)` |
| `PauliY` | 定义一个PauliY逻辑门类 | `layer = PauliY(wires=0)` |
| `PauliZ` | 定义一个PauliZ逻辑门类 | `layer = PauliZ(wires=0)` |
| `X1` | 定义一个X1逻辑门类 | `layer = X1(wires=0)` |
| `RX` | 定义一个RX逻辑门类 | `layer = RX(has_params= True, trainable= True, wires=0)` |
| `RY` | 定义一个RY逻辑门类 | `layer = RY(has_params= True, trainable= True, wires=0)` |
| `RZ` | 定义一个RZ逻辑门类 | `layer = RZ(has_params= True, trainable= True, wires=0)` |
| `CRX` | 定义一个CRX逻辑门类 | `layer = CRX(has_params= True, trainable= True, wires=[0,2])` |
| `CRY` | 定义一个CRY逻辑门类 | `layer = CRY(has_params= True, trainable= True, wires=[0,2])` |
| `CRZ` | 定义一个CRZ逻辑门类 | `layer = CRZ(has_params= True, trainable= True, wires=[0,2])` |
| `U1` | 定义一个U1逻辑门类 | `layer = U1(has_params= True, trainable= True, wires=0)` |
| `U2` | 定义一个U2逻辑门类 | `layer = U2(has_params= True, trainable= True, wires=1)` |
| `U3` | 定义一个U3逻辑门类 | `layer = U3(has_params= True, trainable= True, wires=1)` |
| `CNOT` | 定义一个CNOT逻辑门类,也可称为CX | `layer = CNOT(wires=[0,1])` |
| `CY` | 定义一个CY逻辑门类 | `layer = CY(wires=[0,1])` |
| `CZ` | 定义一个CZ逻辑门类 | `layer = CZ(wires=[0,1])` |
| `CR` | 定义一个CR逻辑门类 | `layer = CR(has_params= True, trainable= True, wires=[0,2])` |
| `SWAP` | 定义一个SWAP逻辑门类 | `layer = SWAP(wires=[0,1])` |
| `CSWAP` | 定义一个CSWAP逻辑门类 | `layer = CSWAP(wires=[0,1,2])` |
| `RXX` | 定义一个RXX逻辑门类 | `layer = RXX(has_params= True, trainable= True, wires=[0,2])` |
| `RYY` | 定义一个RYY逻辑门类 | `layer = RYY(has_params= True, trainable= True, wires=[0,2])` |
| `RZZ` | 定义一个RZZ逻辑门类 | `layer = RZZ(has_params= True, trainable= True, wires=[0,2])` |
| `RZX` | 定义一个RZX逻辑门类 | `layer = RZX(has_params= True, trainable= True, wires=[0,2])` |
| `Toffoli` | 定义一个Toffoli逻辑门类 | `layer = Toffoli(wires=[0,2,1])` |
| `IsingXX` | 定义一个IsingXX逻辑门类 | `layer = IsingXX(has_params= True, trainable= True, wires=[0,2])` |
| `IsingYY` | 定义一个IsingYY逻辑门类 | `layer = IsingYY(has_params= True, trainable= True, wires=[0,2])` |
| `IsingZZ` | 定义一个IsingZZ逻辑门类 | `layer = IsingZZ(has_params= True, trainable= True, wires=[0,2])` |
| `IsingXY` | 定义一个IsingXY逻辑门类 | `layer = IsingXY(has_params= True, trainable= True, wires=[0,2])` |
| `PhaseShift` | 定义一个PhaseShift逻辑门类 | `layer = PhaseShift(has_params= True, trainable= True, wires=1)` |
| `MultiRZ` | 定义一个MultiRZ逻辑门类 | `layer = MultiRZ(has_params= True, trainable= True, wires=[0,2])` |
| `SDG` | 定义一个SDG逻辑门类 | `layer = SDG(wires=0)` |
| `TDG` | 定义一个TDG逻辑门类 | `layer = TDG(wires=0)` |
| `ControlledPhaseShift` | 定义一个ControlledPhaseShift逻辑门类 | `layer = ControlledPhaseShift(has_params= True, trainable= True, wires=[0,2])` |

每个逻辑门示例均遵循同一模式：`set_backend("torch")` → `device.reset_states(batchsize)` → `layer(q_machine=device)` → 读取 `device.states`。以 Hadamard 为例：

**Example:**
```python
from pyvqnet.qnn.vqc.sv.torch import Hadamard,QMachine
import pyvqnet
pyvqnet.backends.set_backend("torch")
device = QMachine(4)
layer = Hadamard(wires=0)
batchsize = 1
device.reset_states(1)
layer(q_machine = device)
print(device.states)
```

**MultiControlledX example:**
```python
import pyvqnet
pyvqnet.backends.set_backend("torch")
from pyvqnet.qnn.vqc.sv.torch import QMachine,MultiControlledX
from pyvqnet.tensor import QTensor,kcomplex64
qm = QMachine(4,dtype=kcomplex64)
qm.reset_states(2)
mcx = MultiControlledX( 
                init_params=None,
                wires=[2,3,0,1],
                dtype=kcomplex64,
                use_dagger=False,control_values=[1,0,0])
y = mcx(q_machine = qm)
print(qm.states)
```

### 测量接口

#### Probability

```python
pyvqnet.qnn.vqc.sv.torch.Probability(wires=None, name="")
```

**Description:**
计算量子线路在特定比特上概率测量结果。继承于 `pyvqnet.qnn.vqc.sv.torch.QModule` 以及 `torch.nn.Module`。

**Parameters:**
- `wires` - 测量比特的索引，列表、元组或者整数
- `name` - 模块的名字，默认 `""`
- **Returns:** 测量结果，QTensor

**Example:**
```python
import pyvqnet
pyvqnet.backends.set_backend("torch")
from pyvqnet.qnn.vqc.sv.torch import Probability,rx,ry,cnot,QMachine,rz
from pyvqnet.tensor import QTensor
from pyvqnet import kfloat64
x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)
qm = QMachine(4)
qm.reset_states(2)
rz(q_machine=qm,wires=0,params=x[:,[0]])
rz(q_machine=qm,wires=1,params=x[:,[0]])
cnot(q_machine=qm,wires=[0,1])
ry(q_machine=qm,wires=2,params=x[:,[1]])
cnot(q_machine=qm,wires=[0,2])
rz(q_machine=qm,wires=3,params=x[:,[1]])
ma = Probability(wires=1)
y =ma(q_machine=qm)
```

#### MeasureAll

```python
pyvqnet.qnn.vqc.sv.torch.MeasureAll(obs=None, name="")
```

**Description:**
计算量子线路的测量结果，支持输入观测量 `obs`。其格式可以为字典格式，用于表示一个由多个 Pauli 算符组合而成的可观测量；列表形式，表示多个期望值的可观测量列表。例如：

- `{'X0': 0.23}` 表示在量子比特 0 上作用 PauliX，系数为 0.23
- `{'X1 Z2':2.4,'Y2':-0.5}` 对应于观测量 `2.4 * X1 @ Z2 - 0.5 * Y2`
- `[{'X1 Z2':4,'Z1 Z0':3},{'X1 Y2 Z0':3.5}]` 对应于两个观测量 `4 * X1 @ Z2 + 3 * Z1 @ Z0` 以及 `3.5 * X1 @ Y2 @ Z0`

**Parameters:**
- `obs` - observable
- `name` - 模块的名字，默认 `""`
- **Returns:** 一个 MeasureAll 测量方法实例

**Example:**
```python
import pyvqnet
pyvqnet.backends.set_backend("torch")
from pyvqnet.qnn.vqc.sv.torch import MeasureAll,rx,ry,cnot,QMachine,rz
from pyvqnet.tensor import QTensor
from pyvqnet import kfloat64
x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)
qm = QMachine(4)
qm.reset_states(2)
rz(q_machine=qm,wires=0,params=x[:,[0]])
rz(q_machine=qm,wires=1,params=x[:,[0]])
cnot(q_machine=qm,wires=[0,1])
ry(q_machine=qm,wires=2,params=x[:,[1]])
cnot(q_machine=qm,wires=[0,2])
rz(q_machine=qm,wires=3,params=x[:,[1]])
obs_list = [{
    "Z0 Z1" :2
}, {
    "X1 X0" :2
}]
ma = MeasureAll(obs = obs_list)
y = ma(q_machine=qm)
print(y)
```

#### Samples

```python
pyvqnet.qnn.vqc.sv.torch.Samples(wires=None, obs=None, shots = 1,name="")
```

**Description:**
获取特定线路上的带有 shot 的样本结果。继承于 `pyvqnet.qnn.vqc.sv.torch.QModule` 以及 `torch.nn.Module`。

**Parameters:**
- `wires` - 样本量子比特索引。默认值 None，根据运行时使用模拟器的所有比特
- `obs` - 该值只能设为 None
- `shots` - 样本重复次数，默认值 1
- `name` - 此模块的名称，默认值 `""`
- **Returns:** 一个 Samples 测量方法实例

**Example:**
```python
import pyvqnet
pyvqnet.backends.set_backend("torch")
from pyvqnet.qnn.vqc.sv.torch import Samples,rx,ry,cnot,QMachine,rz
from pyvqnet.tensor import QTensor
from pyvqnet import kfloat64
x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)

qm = QMachine(4)
qm.reset_states(2)
rz(q_machine=qm,wires=0,params=x[:,[0]])
rx(q_machine=qm,wires=1,params=x[:,[0]])
cnot(q_machine=qm,wires=[0,1])

cnot(q_machine=qm,wires=[0,2])
ry(q_machine=qm,wires=3,params=x[:,[1]])

ma = Samples(wires=[0,1,2],shots=3)
y = ma(q_machine=qm)
print(y)
```

#### HermitianExpval

```python
pyvqnet.qnn.vqc.sv.torch.HermitianExpval(obs=None, name="")
```

**Description:**
计算量子线路某个厄密特量的期望。继承于 `pyvqnet.qnn.vqc.sv.torch.QModule` 以及 `torch.nn.Module`。

**Parameters:**
- `obs` - 厄密特量
- `name` - 模块的名字，默认 `""`
- **Returns:** 一个 HermitianExpval 测量方法实例

**Example:**
```python
import pyvqnet
pyvqnet.backends.set_backend("torch")
from pyvqnet.qnn.vqc.sv.torch import QMachine, rx,ry,\
    RX, RY, CNOT, PauliX, PauliZ, VQC_RotCircuit,HermitianExpval
from pyvqnet.tensor import QTensor, tensor
from pyvqnet.nn import Parameter
import numpy as np
bsz = 3
H = np.array([[8, 4, 0, -6], [4, 0, 4, 0], [0, 4, 8, 0], [-6, 0, 0, 0]])
class QModel(pyvqnet.nn.Module):
    def __init__(self, num_wires, dtype):
        super(QModel, self).__init__()
        self.rot_param = Parameter((3, ))
        self.rot_param.copy_value_from(tensor.QTensor([-0.5, 1, 2.3]))
        self._num_wires = num_wires
        self._dtype = dtype
        self.qm = QMachine(num_wires, dtype=dtype)
        self.rx_layer1 = VQC_RotCircuit
        self.ry_layer2 = RY(has_params=True,
                            trainable=True,
                            wires=0,
                            init_params=tensor.QTensor([-0.5]))
        self.xlayer = PauliX(wires=0)
        self.cnot = CNOT(wires=[0, 1])
        self.measure = HermitianExpval(obs = {'wires':(1,0),'observables':tensor.to_tensor(H)})

    def forward(self, x, *args, **kwargs):
        self.qm.reset_states(x.shape[0])

        rx(q_machine=self.qm, wires=0, params=x[:, [1]])
        ry(q_machine=self.qm, wires=1, params=x[:, [0]])
        self.xlayer(q_machine=self.qm)
        self.rx_layer1(params=self.rot_param, wire=1, q_machine=self.qm)
        self.ry_layer2(q_machine=self.qm)
        self.cnot(q_machine=self.qm)
        rlt = self.measure(q_machine = self.qm)

        return rlt


input_x = tensor.arange(1, bsz * 2 + 1,
                        dtype=pyvqnet.kfloat32).reshape([bsz, 2])
input_x.requires_grad = True

qunatum_model = QModel(num_wires=2, dtype=pyvqnet.kcomplex64)

batch_y = qunatum_model(input_x)
batch_y.backward(pyvqnet.tensor.ones_like(batch_y))
```

### vqc_basisrotation

```python
pyvqnet.qnn.vqc.sv.torch.vqc_basisrotation(q_machine: pyvqnet.qnn.vqc.sv.torch.QMachine, wires, unitary_matrix: QTensor, check=False)
```

**Description:**
实现一个电路，提供可用于执行精确的单体基础旋转的整体。线路来自于 [arXiv:1711.04789](https://arxiv.org/abs/1711.04789) 中给出的单粒子费米子确定的酉变换 `U(u)`，通过使用论文 [Optica, 3, 1460 (2016)](https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743) 中给出的方案。

**Parameters:**
- `q_machine` - 量子虚拟机
- `wires` - 作用的量子位
- `unitary_matrix` - 指定基础变换的矩阵
- `check` - 检测 `unitary_matrix` 是否为酉矩阵

**Example:**
```python
import pyvqnet

pyvqnet.backends.set_backend("torch")
from pyvqnet.qnn.vqc.sv.torch import vqc_basisrotation, QMachine
from pyvqnet.tensor import QTensor
import numpy as np

V = np.array([[0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
            [0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
            [-0.21271 + 0.34938j, -0.38853 + 0.36497j, 0.61467 - 0.41317j]])

eigen_vals, eigen_vecs = np.linalg.eigh(V)
umat = eigen_vecs.T
wires = range(len(umat))

qm = QMachine(len(umat))

vqc_basisrotation(q_machine=qm,
                wires=wires,
                unitary_matrix=QTensor(umat, dtype=qm.states.dtype))

print(qm.states)
```

### 其他 sv.torch 接口

官方文档在 `pyvqnet.qnn.vqc.sv.torch` 下还提供（此处仅列名，签名见官方 RST）：`QuantumLayerAdjoint`、`VQC_HardwareEfficientAnsatz`、`VQC_BasicEntanglerTemplate`、`VQC_StronglyEntanglingTemplate`、`VQC_QuantumEmbedding`、`ExpressiveEntanglingAnsatz`，以及 `vqc_basis_embedding`、`vqc_angle_embedding`、`vqc_amplitude_embedding`、`vqc_iqp_embedding`、`vqc_rotcircuit`、`vqc_crot_circuit`、`vqc_controlled_hadamard`、`vqc_ccz`、`vqc_fermionic_single_excitation`、`vqc_fermionic_double_excitation`、`vqc_uccsd`、`vqc_zfeaturemap`、`vqc_zzfeaturemap`、`vqc_allsinglesdoubles` 等函数接口。

---

## torch 后端张量网络接口 (`pyvqnet.qnn.vqc.tn.torch`)

张量网络（Tensor Network）通过将复杂的张量分解为多个低维张量的网络，显著降低了计算复杂度。矩阵乘积态（Matrix Product State, MPS）是张量网络的一种特殊形式，MPS 将量子态表示为一系列矩阵的乘积，从而有效减少参数数量，降低了计算复杂度。

`pyvqnet.qnn.vqc.tn.torch` 基于 `torch` 后端，对张量网络构建量子线路的功能提供支持，包括构建量子线路基类、量子逻辑门、量子线路以及测量方法，并通过自动微分模拟代替参数移位法计算参数梯度。以 MPS 方式构建量子线路可支持大比特（100 及以上）量子线路模拟。

**Import path:** `from pyvqnet.qnn.vqc.tn.torch import ...`

**Warning:**
- 通过 `TNQMachine` 中 `use_mps` 参数开启 MPS 构建量子线路功能，支持大比特（100 以及以上）量子线路实现
- 批量化与经典模块下使用方式不同，基于 vmap 的方式，数据以及参数构建线路需降一维输入，即对应态矢模拟时代码 `x[:,i]` 需要改为 `x[i]`，批次化执行必须同时基于 `TNQMachine` 和 `TNQModule` 并使用 `TNQMachine` 的 `reset_states` 显式指定批次大小

### TNQModule

```python
pyvqnet.qnn.vqc.tn.torch.TNQModule(use_jit=False,vectorized_argnums=0,name="")
```

**Description:**
在 `torch` 后端下，定义张量网络下量子变分线路模型 `Module` 应该继承的基类。该类用于使用张量网络模块来执行量子线路。基于张量网络编写变分量子线路模型需要继承于 `TNQModule`。

**Parameters:**
- `use_jit` - 开启即时编译功能，默认为 False
- `vectorized_argnums` - 要被向量化的参数，这些参数应该在同一维共享相同的批次形状，默认值为 0
- `name` - 模块名

**Note:**
- 开启 `use_jit` 后，模型会使用 `jax` 的 `jit` 进行即时编译，首次运行将会进行编译，会耗时较长
- 该类以及其派生类仅适用于 `pyvqnet.backends.set_backend("torch")`，不要与默认 `pyvqnet.nn` 下的 `Module` 混用

**Example:**（示例中通过 `qcircuit.ry` / `qcircuit.rz` 函数形式构建线路，并通过 `qmeasure.MeasureAll` 测量）
```python
import pyvqnet
from pyvqnet.nn import Parameter
pyvqnet.backends.set_backend("torch")
from pyvqnet.qnn.vqc.tn.torch import TNQModule
from pyvqnet.qnn.vqc.tn.torch import TNQMachine, RX, RY, CNOT, PauliX, PauliZ,qmeasure,qcircuit,VQC_RotCircuit
class QModel(TNQModule):
    def __init__(self, num_wires, dtype,batch_size=2):
        super(QModel, self).__init__()

        self._num_wires = num_wires
        self._dtype = dtype
        self.qm = TNQMachine(num_wires, dtype=dtype)

        self.w = Parameter((2,4,3),initializer=pyvqnet.utils.initializer.quantum_uniform)
        self.cnot = CNOT(wires=[0, 1])
        self.batch_size = batch_size
    def forward(self, x, *args, **kwargs):
        self.qm.reset_states(batchsize=self.batch_size)

        def get_cnot(nqubits,qm):
            for i in range(len(nqubits) - 1):
                CNOT(wires = [nqubits[i], nqubits[i + 1]])(q_machine = qm)
            CNOT(wires = [nqubits[len(nqubits) - 1], nqubits[0]])(q_machine = qm)


        def build_circuit(weights, xx, nqubits,qm):
            def Rot(weights_j, nqubits,qm):#pylint:disable=invalid-name
                VQC_RotCircuit(qm,nqubits,weights_j)

            def basisstate(qm,xx, nqubits):
                for i in nqubits:
                    qcircuit.rz(q_machine=qm, wires=i, params=xx[i])
                    qcircuit.ry(q_machine=qm, wires=i, params=xx[i])
                    qcircuit.rz(q_machine=qm, wires=i, params=xx[i])

            basisstate(qm,xx,nqubits)

            for i in range(weights.shape[0]):

                weights_i = weights[i, :, :]
                for j in range(len(nqubits)):
                    weights_j = weights_i[j]
                    Rot(weights_j, nqubits[j],qm)
                get_cnot(nqubits,qm)

        build_circuit(self.w, x,range(4),self.qm)

        y= qmeasure.MeasureAll(obs={'Z0': 1})(self.qm)
        return y


x= pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32)
model = QModel(4,pyvqnet.kcomplex64,2)
y = model(x)
y.backward(pyvqnet.tensor.ones_like(y))
```

### TNQMachine

```python
pyvqnet.qnn.vqc.tn.torch.TNQMachine(num_wires, dtype=pyvqnet.kcomplex64,use_mps=False)
```

**Description:**
变分量子计算的模拟器类，包含 `states` 属性为量子线路的 statevectors。基于张量网络编写变分量子线路设备需要 `TNQMachine` 进行初始化。该类继承于 `pyvqnet.nn.tn.TorchModule` 以及 `pyvqnet.qnn.QMachine`，可以作为 `torch.nn.Module` 的一个子模块加入 `TNQModule` 的模型中。

**Warning:**
- 在每次运行一个完整的量子线路之前，必须使用 `pyvqnet.qnn.vqc.tn.torch.TNQMachine.reset_states(batchsize)` 将模拟器里面初态重新初始化，并且广播为 `(batchsize,*)` 维度从而适应批量数据训练
- 在张量网络的量子线路中，默认会开启 `vmap` 功能，在线路上的逻辑门参数上均为舍弃了批次维度，使用时，调用参数若维度为 `[batch_size, *]`，在使用时舍弃第一个 batch_size 维度，直接使用后面维度，如对输入数据 `x[:,1]` -> `x[1]`，对可训练参数也一致

**Parameters:**
- `num_wires` - 量子比特数
- `dtype` - 计算数据的数据类型。默认值是 `pyvqnet.kcomplex64`，对应的参数精度为 `pyvqnet.kfloat32`
- `use_mps` - 是否基于 mpscircuit 进行模拟，用于模拟大比特量子线路执行

**Methods:**

```python
TNQMachine.get_states()
```

获得张量网络中的 states。

官方文档中 `TNQMachine` 的示例与上方 `TNQModule` 的示例相同（`QModel` 中同时演示了 `TNQMachine` 的构建与 `reset_states` 用法）。

单门快速验证模式与态矢接口一致（`set_backend("torch")` → `device.reset_states(batchsize)` → `layer(q_machine=device)` → `device.get_states()`）：

**Example:**
```python
import pyvqnet
pyvqnet.backends.set_backend("torch")
from pyvqnet.qnn.vqc.tn.torch import Hadamard,TNQMachine
device = TNQMachine(4)
layer = Hadamard(wires=0)
batchsize = 1
device.reset_states(1)
layer(q_machine = device)
print(device.get_states())
```

### 变分量子逻辑门

以下量子线路模块继承于 `pyvqnet.qnn.vqc.tn.torch.TNQModule`，其中计算使用 `torch.Tensor` 进行计算。`pyvqnet.qnn.vqc.tn` 下的函数接口直接支持 `torch` 后端的 `QTensor` 进行计算。

所有逻辑门类共享如下签名（以 `I` 为例，其余仅类名不同）：

```python
pyvqnet.qnn.vqc.tn.torch.I(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.Hadamard(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.T(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.S(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.PauliX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.PauliY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.PauliZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.RX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.RY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.RZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.CRX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.CRY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.CRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.U1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.U2(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.U3(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.CNOT(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.CY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.CZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.CR(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.SWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.CSWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.RXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.RYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.RZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.RZX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.Toffoli(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.IsingXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.IsingYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.IsingZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.IsingXY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.PhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.MultiRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.SDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.TDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
pyvqnet.qnn.vqc.tn.torch.ControlledPhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
```

**Parameters (all gates):** 与 `pyvqnet.qnn.vqc.sv.torch` 逻辑门一致：`has_params` / `trainable` / `init_params` / `wires` / `dtype` / `use_dagger`（含义与默认值见上一节）。

**Gate list and example instantiation (from official doc):**

| Gate | Description | Example instantiation |
|------|-------------|----------------------|
| `I` | 定义一个I逻辑门类 | `layer = I(wires=0)` |
| `Hadamard` | 定义一个Hadamard逻辑门类 | `layer = Hadamard(wires=0)` |
| `T` | 定义一个T逻辑门类 | `layer = T(wires=0)` |
| `S` | 定义一个S逻辑门类 | `layer = S(wires=0)` |
| `PauliX` | 定义一个PauliX逻辑门类 | `layer = PauliX(wires=0)` |
| `PauliY` | 定义一个PauliY逻辑门类 | `layer = PauliY(wires=0)` |
| `PauliZ` | 定义一个PauliZ逻辑门类 | `layer = PauliZ(wires=0)` |
| `RX` | 定义一个RX逻辑门类 | `layer = RX(has_params= True, trainable= True, wires=0)` |
| `RY` | 定义一个RY逻辑门类 | `layer = RY(has_params= True, trainable= True, wires=0)` |
| `RZ` | 定义一个RZ逻辑门类 | `layer = RZ(has_params= True, trainable= True, wires=0)` |
| `CRX` | 定义一个CRX逻辑门类 | `layer = CRX(has_params= True, trainable= True, wires=[0,2])` |
| `CRY` | 定义一个CRY逻辑门类 | `layer = CRY(has_params= True, trainable= True, wires=[0,2])` |
| `CRZ` | 定义一个CRZ逻辑门类 | `layer = CRZ(has_params= True, trainable= True, wires=[0,2])` |
| `U1` | 定义一个U1逻辑门类 | `layer = U1(has_params= True, trainable= True, wires=0)` |
| `U2` | 定义一个U2逻辑门类 | `layer = U2(has_params= True, trainable= True, wires=0)` |
| `U3` | 定义一个U3逻辑门类 | `layer = U3(has_params= True, trainable= True, wires=0)` |
| `CNOT` | 定义一个CNOT逻辑门类,也可称为CX | `layer = CNOT(wires=[0,1])` |
| `CY` | 定义一个CY逻辑门类 | `layer = CY(wires=[0,1])` |
| `CZ` | 定义一个CZ逻辑门类 | `layer = CZ(wires=[0,1])` |
| `CR` | 定义一个CR逻辑门类 | `layer = CR(has_params= True, trainable= True, wires=[0,2])` |
| `SWAP` | 定义一个SWAP逻辑门类 | `layer = SWAP(wires=[0,1])` |
| `CSWAP` | 定义一个CSWAP逻辑门类 | `layer = CSWAP(wires=[0,1,2])` |
| `RXX` | 定义一个RXX逻辑门类 | `layer = RXX(has_params= True, trainable= True, wires=[0,2])` |
| `RYY` | 定义一个RYY逻辑门类 | `layer = RYY(has_params= True, trainable= True, wires=[0,2])` |
| `RZZ` | 定义一个RZZ逻辑门类 | `layer = RZZ(has_params= True, trainable= True, wires=[0,2])` |
| `RZX` | 定义一个RZX逻辑门类 | `layer = RZX(has_params= True, trainable= True, wires=[0,2])` |
| `Toffoli` | 定义一个Toffoli逻辑门类 | `layer = Toffoli(wires=[0,2,1])` |
| `IsingXX` | 定义一个IsingXX逻辑门类 | `layer = IsingXX(has_params= True, trainable= True, wires=[0,2])` |
| `IsingYY` | 定义一个IsingYY逻辑门类 | `layer = IsingYY(has_params= True, trainable= True, wires=[0,2])` |
| `IsingZZ` | 定义一个IsingZZ逻辑门类 | `layer = IsingZZ(has_params= True, trainable= True, wires=[0,2])` |
| `IsingXY` | 定义一个IsingXY逻辑门类 | `layer = IsingXY(has_params= True, trainable= True, wires=[0,2])` |
| `PhaseShift` | 定义一个PhaseShift逻辑门类 | `layer = PhaseShift(has_params= True, trainable= True, wires=1)` |
| `MultiRZ` | 定义一个MultiRZ逻辑门类 | `layer = MultiRZ(has_params= True, trainable= True, wires=[0,2])` |
| `SDG` | 定义一个SDG逻辑门类 | `layer = SDG(wires=0)` |
| `TDG` | 定义一个TDG逻辑门类 | `layer = TDG(wires=0)` |
| `ControlledPhaseShift` | 定义一个ControlledPhaseShift逻辑门类 | `layer = ControlledPhaseShift(has_params= True, trainable= True, wires=[0,2])` |

注意：tn.torch 后端不提供 `X1` 与 `MultiControlledX` 门（相比 sv.torch）。

### MeasureAll

```python
pyvqnet.qnn.vqc.tn.torch.MeasureAll(obs=None, name="")
```

**Description:**
计算量子线路的测量结果，支持输入 obs 为多个或单个泡利算子或哈密顿量。例如：

- `{'wires': [0, 1], 'observables': ['x', 'i'],'coefficient':[0.23,-3.5]}`
- `{'X0': 0.23}`
- `[{'wires': [0, 2, 3],'observables': ['X', 'Y', 'Z'],'coefficient': [1, 0.5, 0.4]}, {'wires': [0, 1, 2],'observables': ['X', 'Y', 'Z'],'coefficient': [1, 0.5, 0.4]}]`

该类继承于 `pyvqnet.qnn.vqc.tn.torch.QModule` 以及 `torch.nn.Module`，可以作为 `torch.nn.Module` 的一个子模块加入 torch 的模型中。

**Parameters:**
- `obs` - observable
- `name` - 模块的名字，默认 `""`
- **Returns:** 一个 MeasureAll 测量方法实例

**Example:**
```python
import pyvqnet
pyvqnet.backends.set_backend("torch")
from pyvqnet.qnn.vqc.tn.torch import MeasureAll, qcircuit, TNQMachine, Hadamard, CNOT
qm = TNQMachine(2)
qm.reset_states(2)
Hadamard(wires=0)(q_machine=qm)
CNOT(wires=[0, 1])(q_machine=qm)
obs_list = [{"Z0 Z1": 2}, {"X1 X0": 2}]
ma = MeasureAll(obs=obs_list)
y = ma(q_machine=qm)
print(y)
```

### vqc_basisrotation

```python
pyvqnet.qnn.vqc.tn.vqc_basisrotation(q_machine: pyvqnet.qnn.vqc.tn.torch.TNQMachine, wires, unitary_matrix: QTensor, check=False)
```

**Description:**
实现一个电路，提供可用于执行精确的单体基础旋转的整体。线路来自于 [arXiv:1711.04789](https://arxiv.org/abs/1711.04789) 中给出的单粒子费米子确定的酉变换 `U(u)`，通过使用论文 [Optica, 3, 1460 (2016)](https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743) 中给出的方案。

**Parameters:**
- `q_machine` - 量子虚拟机
- `wires` - 作用的量子位
- `unitary_matrix` - 指定基础变换的矩阵
- `check` - 检测 `unitary_matrix` 是否为酉矩阵

**Example:**
```python
import pyvqnet

pyvqnet.backends.set_backend("torch")
from pyvqnet.qnn.vqc.tn.torch import vqc_basisrotation, TNQMachine
from pyvqnet.tensor import QTensor
import numpy as np

V = np.array([[0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
            [0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
            [-0.21271 + 0.34938j, -0.38853 + 0.36497j, 0.61467 - 0.41317j]])

eigen_vals, eigen_vecs = np.linalg.eigh(V)
umat = eigen_vecs.T
wires = range(len(umat))

qm = TNQMachine(len(umat))

vqc_basisrotation(q_machine=qm,
                wires=wires,
                unitary_matrix=QTensor(umat, dtype=qm.dtype))

print(qm.get_states())
```

### 其他 tn.torch 接口

官方文档在 `pyvqnet.qnn.vqc.tn.torch` / `pyvqnet.qnn.vqc.tn` 下还提供（此处仅列名，签名见官方 RST）：测量类 `Probability`、`Samples`、`HermitianExpval`，测量函数 `VQC_Purity`、`VQC_VarMeasure`、`VQC_DensityMatrixFromQstate`，模板类 `VQC_HardwareEfficientAnsatz`、`VQC_BasicEntanglerTemplate`、`VQC_StronglyEntanglingTemplate`、`VQC_QuantumEmbedding`、`ExpressiveEntanglingAnsatz`，以及 `vqc_basis_embedding`、`vqc_angle_embedding`、`vqc_iqp_embedding`、`vqc_rotcircuit`、`vqc_crot_circuit`、`vqc_controlled_hadamard`、`vqc_ccz`、`vqc_fermionic_single_excitation`、`vqc_fermionic_double_excitation`、`vqc_uccsd`、`vqc_zfeaturemap`、`vqc_zzfeaturemap`、`vqc_allsinglesdoubles` 等函数接口。

**Dependencies (TN backend):**

本模块基于 `jax` 实现自动微分与 GPU 加速。默认安装 `pyvqnet` 不包含该依赖：

```bash
# CPU
pip install jax
# GPU（需 CUDA 12.6）
pip install "jax[cuda12]"
# 此外还需额外安装 tensornetwork
pip install tensornetwork
```

---

## `pyvqnet.qnn.pq3.torch` 量子层

使用 pyqpanda3 进行线路计算的训练变分量子线路接口。以下接口的量子计算部分使用 pyqpanda3 <https://qcloud.originqc.com.cn/document/qpanda-3/index.html>。

**Warning:** 需要安装最新版本 pyqpanda3。

### TorchQcloud3QuantumLayer

```python
pyvqnet.qnn.pq3.torch.qpanda3_layer.TorchQcloud3QuantumLayer(origin_qprog_func, qcloud_token, para_num, pauli_str_dict=None, shots = 1000, initializer=None, dtype=None, name="", diff_method="parameter_shift", submit_kwargs={}, query_kwargs={})
```

**Description:**
使用 pyqpanda3 的本源量子 <https://qcloud.originqc.com.cn/> 真实芯片的抽象计算模块。它提交参数化量子电路到真实芯片并获得测量结果。如果 `diff_method == "random_coordinate_descent"`，该层将随机选择单个参数来计算梯度，其他参数将保持为零。参考：<https://arxiv.org/abs/2311.00088>

**Note:**
- `qcloud_token` 为您到 <https://qcloud.originqc.com.cn/> 中申请的 API token
- `origin_qprog_func` 需返回 `pyqpanda3.core.QProg` 类型的数据，如果没有设置测量观测量 `pauli_str_dict`，需要保证该 QProg 中已经插入了 measure
- `origin_qprog_func` 的形式必须为 `origin_qprog_func(input, param)`：`input` 输入 1~2 维经典数据（二维时第一个维度为批处理大小）；`param` 输入一维的变分量子线路的待训练参数

**Warning:**
该类继承于 `pyvqnet.nn.Module` 以及 `torch.nn.Module`，可以作为 `torch.nn.Module` 的一个子模块加入 torch 的模型中。该类的 `_buffers` 中的数据为 `torch.Tensor` 类型，`_parameters` 中的数据为 `torch.nn.Parameter` 类型。

**Parameters:**
- `origin_qprog_func` - QPanda 构建的变分量子电路函数，必须返回 QProg
- `qcloud_token` - `str` - 量子机的类型或用于执行的云令牌
- `para_num` - `int` - 参数数量，参数是大小为 `[para_num]` 的 QTensor
- `pauli_str_dict` - `dict|list` - 表示量子电路中泡利运算符的字典或字典列表。默认为 None，则进行测量操作；如果输入泡利算符的字典，则会计算单个期望或者多个期望
- `shots` - `int` - 测量次数。默认值为 1000
- `initializer` - 参数值的初始化器。默认为 None，使用 0~2*pi 正态分布
- `dtype` - 参数的数据类型。默认值为 None，即使用默认数据类型 `pyvqnet.kfloat32`
- `name` - 模块的名称。默认为空字符串
- `diff_method` - 梯度计算的微分方法。默认为 `"parameter_shift"`，或 `"random_coordinate_descent"`
- `submit_kwargs` - 用于提交量子电路的附加关键字参数，默认 `{"if_print_qcloud_log":False,"chip_id":"WK_C180","is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":False,"server_ip_address":"","use_qwc":True}`，当设置 `test_qcloud_fake` 为 True 则本地 CPUQVM 模拟
- `query_kwargs` - 用于查询量子结果的附加关键字参数，默认 `{"timeout":2,"print_query_info":True,"sub_circuits_split_size":1}`

**Example:**
```python
import pyqpanda3.core as pq
import pyvqnet
from pyvqnet.qnn.vqc.sv.torch import TorchQcloud3QuantumLayer

pyvqnet.backends.set_backend("torch")
def qfun(input,param):

    m_qlist = range(6)
    cbits = range(6)
    measure_qubits = [0,2]
    m_prog = pq.QProg()
    cir = pq.QCircuit()
    cir<<pq.RZ(m_qlist[0],input[0])
    cir<<pq.CNOT(m_qlist[0],m_qlist[1])
    cir<<pq.RY(m_qlist[1],param[0])
    cir<<pq.CNOT(m_qlist[0],m_qlist[2])
    cir<<pq.RZ(m_qlist[1],input[1])
    cir<<pq.RY(m_qlist[2],param[1])
    cir<<pq.H(m_qlist[2])
    m_prog<<cir

    for idx, ele in enumerate(measure_qubits):
        m_prog << pq.measure(m_qlist[ele], cbits[idx])  # pylint: disable=expression-not-assigned
    return m_prog

l = TorchQcloud3QuantumLayer(qfun,
                "your_api_token",
                2,
                pauli_str_dict=None,
                shots = 1000,
                initializer=None,
                dtype=None,
                name="",
                diff_method="parameter_shift",
                submit_kwargs={"test_qcloud_fake":True},
                query_kwargs={})
x = pyvqnet.tensor.QTensor([[0.56,1.2],[0.56,1.2],[0.56,1.2],[0.56,1.2],[0.56,1.2]],requires_grad= True)
y = l(x)
print(y)
y.backward(pyvqnet.tensor.ones_like(y))
print(l.m_para.grad)
print(x.grad)
```

### TorchQpanda3QuantumLayer

```python
pyvqnet.qnn.pq3.torch.qpanda3_layer.TorchQpanda3QuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")
```

**Description:**
变分量子层的抽象计算模块。对一个参数化的量子线路使用 pyqpanda3 进行仿真，得到测量结果。该变分量子层继承了 VQNet 框架的梯度计算模块，可以使用参数移位法等计算线路参数的梯度，训练变分量子线路模型或将变分量子线路嵌入混合量子和经典模型。如您更加熟悉 pyqpanda3 语法，可以使用该接口。

**Parameters:**
- `qprog_with_measure` - 用 pyqpanda3 构建的量子线路运行和测量函数
- `para_num` - `int` - 参数个数
- `diff_method` - 求解量子线路参数梯度的方法，`"parameter_shift"` 或 `"finite_diff"`，默认为 `"parameter_shift"`
- `delta` - 有限差分计算梯度时的 δ
- `dtype` - 参数的数据类型，默认 None，使用默认数据类型 `kfloat32`，代表 32 位浮点数
- `name` - 这个模块的名字，默认为 `""`
- **Returns:** 一个可以计算量子线路的模块

**Note:**
`qprog_with_measure` 是 pyQPanda 中定义的量子线路函数（参见 <https://qcloud.originqc.com.cn/document/qpanda-3/db/d6c/tutorial_circuit_and_program.html>）。此函数必须包含以下参数作为函数入参（即使某个参数未实际使用），否则无法在本函数中正常运行：`input` 输入一维经典数据，如果没有输入 None；`param` 输入一维的变分量子线路的待训练参数。

**Example:**
```python
import pyqpanda3.core as pq
from pyvqnet.qnn.pq3 import ProbsMeasure
import numpy as np
from pyvqnet.tensor import QTensor
import pyvqnet
pyvqnet.backends.set_backend("torch")
from pyvqnet.qnn.vqc.sv.torch import TorchQpanda3QuantumLayer
def pqctest (input,param):
    num_of_qubits = 4

    m_machine = pq.CPUQVM()# outside

    qubits =range(num_of_qubits)

    circuit = pq.QCircuit()
    circuit<<pq.H(qubits[0])
    circuit<<pq.H(qubits[1])
    circuit<<pq.H(qubits[2])
    circuit<<pq.H(qubits[3])

    circuit<<pq.RZ(qubits[0],input[0])
    circuit<<pq.RZ(qubits[1],input[1])
    circuit<<pq.RZ(qubits[2],input[2])
    circuit<<pq.RZ(qubits[3],input[3])

    circuit<<pq.CNOT(qubits[0],qubits[1])
    circuit<<pq.RZ(qubits[1],param[0])
    circuit<<pq.CNOT(qubits[0],qubits[1])

    circuit<<pq.CNOT(qubits[1],qubits[2])
    circuit<<pq.RZ(qubits[2],param[1])
    circuit<<pq.CNOT(qubits[1],qubits[2])

    circuit<<pq.CNOT(qubits[2],qubits[3])
    circuit<<pq.RZ(qubits[3],param[2])
    circuit<<pq.CNOT(qubits[2],qubits[3])

    prog = pq.QProg()
    prog<<circuit

    rlt_prob = ProbsMeasure(m_machine,prog,[0,2])
    return rlt_prob

pqc = TorchQpanda3QuantumLayer(pqctest,3)

#classic data as input
input = QTensor([[1.0,2,3,4],[4,2,2,3],[3,3,2,2]],requires_grad=True)

#forward circuits
rlt = pqc(input)

print(rlt)

grad = pyvqnet.tensor.ones(rlt.data.shape)*1000
#backward circuits
rlt.backward(grad)

print(pqc.m_para.grad)
print(input.grad)
```

**Note:** 2.18.1 版本 changelog 中还提及 `TorchQpandaQuantumLayer`（对接 qpanda 模拟器的 pq3 torch 量子层），但其 API 参考签名未收录于 2.18.1 的 `torch_api.rst` 文档中，此处不作记载。

---

## Torch Backend Utilities (`pyvqnet.torch`)

```python
from pyvqnet.torch import *
```

### Type Conversion

```python
pyvqnet.torch.dtype_map_torch(vqnet_dtype)    # QTensor dtype → torch.dtype
pyvqnet.torch.device_map_torch(vqnet_device)   # VQNet device → torch.device
pyvqnet.torch.get_vqnet_dtype(dtype)           # torch.dtype → VQNet dtype
pyvqnet.torch.get_vqnet_device(torch_device)   # torch.device → VQNet device
```

### Gradient Control

```python
pyvqnet.torch.set_grad_enabled(flag: bool)
pyvqnet.torch.get_grad_enabled() -> bool
```

### Random Seed

```python
pyvqnet.torch.set_random_seed(seed: int)
pyvqnet.torch.get_random_seed() -> int
```

### Data Preprocessing

```python
pyvqnet.torch.unary_operators_preprocess(data)   # Ensure data is torch.Tensor
pyvqnet.torch._binary_operators_preprocess(d1, d2)   # Ensure both are torch.Tensor
pyvqnet.torch._multi_operators_preprocess(datas)      # Ensure all are torch.Tensor
```

### Backend Classes

```python
pyvqnet.torch.TorchWrapperBackend      # "torch" backend (QTensor wrapping torch.Tensor)
pyvqnet.torch.TorchNativeBackend       # "torch-native" backend (direct torch.Tensor)
```

### Distributed

```python
from pyvqnet.torch._distributed import *
```

Torch distributed backend functions for multi-GPU training. See `distributed.md` for details.

---

## Torch Initializer (`pyvqnet.torch.initializer`)

```python
from pyvqnet.torch.initializer import he_uniform, zeros, ...
```

Custom initializer functions compatible with torch backend for parameter initialization.

---

## Complete Example: Hybrid Training Loop

```python
import pyvqnet
import torch
from pyvqnet.tensor import QTensor
from pyvqnet.nn.torch import TorchModule, Linear, ReLu, CrossEntropyLoss
from pyvqnet import kint64

# Switch to PyTorch backend
pyvqnet.backends.set_backend("torch")

class Classifier(TorchModule):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 256)
        self.relu = ReLu()
        self.fc2 = Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = Classifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = CrossEntropyLoss()

# Training loop — pure PyTorch
for epoch in range(10):
    # x: QTensor wrapping torch.Tensor, y: QTensor with dtype=kint64
    logits = model(x)
    loss = loss_fn(y, logits)  # VQNet order: (target, output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
