---
name: vqnet2_api
description: PyVQNet/VQNet2.0 API documentation and assistance for quantum machine learning. Use when working with VQNet2.0, QTensor, quantum neural networks, variational quantum circuits (VQC), PyTorch backend integration, quantum convolution (QConv), quantum linear layers (QLinear), or any VQNet API questions. Also triggered by Chinese terms VQNet, QTensor, 变分量子线路, 量子神经网络, 量子机器学习.
license: Apache License 2.0
---

This skill provides expert assistance for quantum machine learning development using **VQNet 2.0/PyVQNet** with pyqpanda3. It helps with API usage, code generation, debugging, and best practices based on the official documentation.

## 🧠 Core Principles (Always Active)

1.  **Framework Version**: Use **VQNet 2.0 + pyqpanda3** syntax. This is the modern combination - do NOT use legacy pyqpanda API.
2.  **Backend Selection**:
    *   Default: `pyvqnet-ad` (C++ accelerated autograd)
    *   Use `torch` backend when integrating with PyTorch models
    *   Use `pyvqnet` for Python autograd (rarely needed)
3.  **QTensor First**: Always use QTensor for tensor operations - it's the native tensor type in VQNet.
4.  **Gradients**: VQNet supports automatic differentiation - ensure `requires_grad=True` for trainable parameters.

## 🚦 Execution Workflow

### Step 1: Intent Analysis & Resource Loading

Analyze the user's request and load the specific resource file using the `Read` tool immediately.

| User Intent | Resource to Read |
| :--- | :--- |
| **QTensor questions** | `references/qtensor.md` - QTensor class, attributes, methods, creation |
| **Quantum layers** | `references/quantum_layers.md` - QuantumLayer, QpandaQProgVQCLayer, QuantumBatchAsyncQcloudLayer, QuantumLayerAdjoint |
| **Quantum templates/gates** | `references/quantum_templates.md` - Embedding circuits, ansatz, quantum gates, single/two-qubit operations |
| **Measurement functions** | `references/measurement.md` - expval, ProbsMeasure, QuantumMeasure, entropy, purity |
| **Classical NN layers** | `references/classic_nn.md` - QLinear, QConv - quantum-classical hybrid neural networks |
| **PyTorch backend** | `references/torch_api.md` - set_backend, get_backend, PyTorch integration |
| **Utility functions** | `references/utils.md` - random seeds, initializers, data utilities |
| **Variational Quantum Circuits** | `references/vqc.md` - VQC implementation, training examples |
| **Quantum Neural Networks** | `references/qnn.md` - QNN architecture, best practices |

### Step 2: Implementation Strategy

1.  **Check Backend**: Determine if user is using the default pyvqnet backend or PyTorch backend.
2.  **Check Types**: Ensure QTensor is used correctly with proper `requires_grad` settings.
3.  **Drafting**: Write code using the patterns from the reference documentation.
4.  **Verify**: Double-check parameter orders and function signatures against the API docs.

### Step 3: Validation

Before outputting code, verify:

*   [ ] Is the import path correct (e.g., `pyvqnet.qnn.pq3.quantumlayer` vs `pyvqnet.tensor`)?
*   [ ] Are quantum layer parameters in the correct order (function first, then para_num)?
*   [ ] Does the quantum circuit function have the correct signature `(input, param)`?
*   [ ] Is `requires_grad` set correctly for trainable parameters?
*   [ ] Are API keys handled correctly (use environment variables for QCloud)?
*   [ ] Does the example include a complete, runnable code snippet?

## 🛠 Quick Reference - Common Patterns

### Basic QTensor Creation

```python
from pyvqnet.tensor import QTensor
from pyvqnet.dtype import *

# Create from list
t1 = QTensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

# Create from numpy
import numpy as np
t2 = QTensor(np.ones([2, 3]), dtype=kfloat64)

print(t1.shape)   # [2, 2]
print(t1.ndim)    # 2
print(t1.size)    # 4
```

### Backend Switching

```python
import pyvqnet
# Switch to PyTorch backend
pyvqnet.backends.set_backend("torch")

# Get current backend
current = pyvqnet.backends.get_backend()
```

### QuantumLayer Example

```python
from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
from pyvqnet.qnn.pq3.measure import ProbsMeasure
from pyvqnet.tensor import QTensor
import pyqpanda3.core as pq

def circuit(input, param):
    num_qubits = 4
    machine = pq.CPUQVM()
    qubits = range(num_qubits)
    circuit = pq.QCircuit()

    # Encode input data
    for i in range(num_qubits):
        circuit << pq.H(i) << pq.RZ(i, input[i])

    # Variational part
    for i in range(num_qubits - 1):
        circuit << pq.CNOT(i, i+1) << pq.RZ(i+1, param[i])

    prog = pq.QProg()
    prog << circuit
    return ProbsMeasure(machine, prog, [0, 1, 2, 3])

layer = QuantumLayer(circuit, 3)  # 3 parameters
x = QTensor([[0.1, 0.2, 0.3, 0.4]])
output = layer(x)
output.backward()
```

## ⚠️ Critical Anti-Patterns (Do NOT Do)

* Never mix backend types - QTensor created in different backends cannot be used together
* Never forget that the quantum circuit function for QuantumLayer **must** have `(input, param)` as parameters
* Never hardcode QCloud API tokens - use environment variables
* Never mix pyqpanda (legacy) and pyqpanda3 imports
* Don't forget to call `backward()` for gradients when training

## 📚 Supported Modules

| Module | Purpose |
|--------|---------|
| `pyvqnet.tensor` | QTensor - core tensor data structure with autograd |
| `pyvqnet.qnn.pq3.quantumlayer` | Quantum layers: QuantumLayer, QpandaQProgVQCLayer, QuantumBatchAsyncQcloudLayer, QuantumLayerAdjoint |
| `pyvqnet.qnn.pq3.measure` | Measurement functions: expval, ProbsMeasure, QuantumMeasure, entropy calculations |
| `pyvqnet.qnn.pq3.template` | Circuit templates: embedding, ansatz, quantum gate combinations |
| `pyvqnet.qnn.pq3.ansatz` | Hardware-efficient ansatz and templates |
| `pyvqnet.qnn.qlinear` | Quantum fully-connected layer |
| `pyvqnet.qnn.qcnn.qconv` | Quantum convolution layer |
| `pyvqnet.backends` | PyTorch backend switching |
| `pyvqnet.utils` | Utilities: random seeds, initializers |
| `pyvqnet.nn` | Classical neural network layers (like PyTorch) |

## 🔖 Backend Comparison

| Backend | Description | Use Case |
|---------|-------------|----------|
| `pyvqnet-ad` | C++ core, C++ autograd (default) | Most cases - best performance |
| `pyvqnet` | C++ core, Python autograd | Debugging |
| `torch` | PyTorch core, PyTorch autograd | Integrating with PyTorch models |
| `torch-native` | Direct PyTorch output | Maximum PyTorch compatibility |

---

**Version**: 1.0
**Compatibility**: VQNet 2.0+ with pyqpanda3
**Last Updated**: 2026
