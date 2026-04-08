# Classical-Quantum Hybrid Neural Network Layers

## QLinear

```python
pyvqnet.qnn.qlinear.QLinear(
    input_channels,
    output_channels,
    machine: str = "CPU"
)
```

**Description:**
Quantum fully-connected (linear) layer. Encodes data into quantum states, evolves through quantum circuit, and measures to get the output.

**Note:** This layer does **not** have trainable variational quantum parameters - it's just the quantum encoding.

**Parameters:**
- `input_channels` - `int` - Number of input channels
- `output_channels` - `int` - Number of output channels
- `machine` - `str` - Virtual machine type ("CPU" or "GPU"), default: "CPU"
- **Returns:** Quantum fully-connected layer

**Example:**
```python
from pyvqnet.tensor import QTensor
from pyvqnet.qnn.qlinear import QLinear

params = [
    [0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
    [1.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
    [1.37454012, 1.95071431, 0.73199394, 1.59865848, 0.15601864, 0.15599452],
    [1.37454012, 1.95071431, 1.73199394, 1.59865848, 0.15601864, 0.15599452]
]
m = QLinear(6, 2)
input = QTensor(params, requires_grad=True)
output = m(input)
output.backward()
print(output)
# [[0.0568473, 0.1264389],
#  [0.1524036, 0.1264389],
#  [0.1524036, 0.1442845],
#  [0.1524036, 0.1442845]]
```

---

## QConv (Quantum Convolution)

```python
pyvqnet.qnn.qcnn.qconv.QConv(
    input_channels,
    output_channels,
    quantum_number,
    stride=(1, 1),
    padding=(0, 0),
    kernel_initializer=normal,
    machine: str = "CPU",
    dtype=None,
    name = ""
)
```

**Description:**
Quantum convolution layer. Uses a quantum circuit to replace the convolution kernel, where each kernel position requires the same number of qubits as the kernel size, encodes data into quantum states, evolves through entanglement and measurement to produce the output.

Reference: *Samuel et al. (2020)* https://arxiv.org/abs/2012.12177

**Algorithm:**
1. Allocate one qubit per input element in the convolution kernel
2. Encode data into quantum states using RY/RZ rotations
3. Entangle qubits using Z and U3 to exchange information
4. Measure to get convolution output

**Parameters:**
- `input_channels` - `int` - Number of input channels
- `output_channels` - `int` - Number of output channels
- `quantum_number` - `int` - Kernel size (number of qubits per kernel)
- `stride` - `tuple` - Stride (height, width), default: `(1, 1)`
- `padding` - `tuple` - Padding (height, width), default: `(0, 0)`
- `kernel_initializer` - Initializer callable for weights, default: `normal` (normal distribution)
- `machine` - `str` - "CPU" or "GPU" simulator, default: "CPU"
- `dtype` - Parameter data type, default: `None` (kfloat32)
- `name` - Layer name, default: ""
- **Returns:** Quantum convolution layer

**Example:**
```python
from pyvqnet.tensor import tensor
from pyvqnet.qnn.qcnn.qconv import QConv

x = tensor.ones([1, 3, 4, 4])
layer = QConv(
    input_channels=3,
    output_channels=2,
    quantum_number=4,
    stride=(2, 2)
)
y = layer(x)
print(y)
# [[[[-0.0889078, -0.0889078],
#   [-0.0889078, -0.0889078]],
#  [[0.7992646, 0.7992646],
#   [0.7992646, 0.7992646]]]
```
