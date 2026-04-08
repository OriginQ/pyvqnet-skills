# VQNet 2.0 AI Skill

AI assistant skill for VQNet 2.0/PyVQNet quantum machine learning development.

## Overview

This skill provides AI assistants (Claude, Cline, etc.) with complete API reference for VQNet 2.0, extracted from the official reStructuredText documentation. It helps with:

- **API lookup** - Quick access to function signatures, parameters, and examples
- **Code generation** - Generate correct VQNet code following official patterns
- **Debugging** - Identify common mistakes (wrong parameter order, backend mixing, etc.)
- **Best practices** - Follow the official documentation conventions

## Supported APIs

- **QTensor** - Core tensor with automatic differentiation
- **Quantum Layers** - `QuantumLayer`, `QpandaQProgVQCLayer`, `QuantumBatchAsyncQcloudLayer`, `QuantumLayerAdjoint`
- **Circuit Templates** - AmplitudeEmbedding, AngleEmbedding, IQPEmbedding, HardwareEfficientAnsatz, and more
- **Measurement** - `expval`, `ProbsMeasure`, `QuantumMeasure`, entropy, purity, mutual information
- **Hybrid Neural Networks** - `QLinear` (quantum fully-connected), `QConv` (quantum convolution)
- **PyTorch Integration** - Backend switching, mixed PyTorch/VQNet development
- **Utilities** - Random seeds, parameter initializers

## Installation

### For Claude Code (Claude CLI)

Claude Code supports skills through the skill system. Clone this repository to your skills directory:

```bash
# Clone into your Claude Code skills directory
git clone https://github.com/Origin-Quantum/vqnet2-skill.git ~/.claude/skills/vqnet2-api
```

Or if you already have this repository locally:

```bash
# Copy/move to your Claude Code skills folder
mkdir -p ~/.claude/skills
cp -r /path/to/vqnet2-skill ~/.claude/skills/vqnet2-api
```

### For VS Code Cline / Roo Code

Cline supports custom instructions and external skill packages. Add this repository as a skill package:

1. In VS Code, open Cline settings
2. Find "Custom Instructions" or "Skill Packages" section
3. Add the path to this cloned repository:
   ```
   /path/to/vqnet2-skill
   ```

### Manual Installation (Any AI Assistant)

If your AI assistant doesn't have a skill system, you can:

1. Clone this repository
2. When asking about VQNet, reference the relevant documentation files from `references/` directory

```bash
git clone https://github.com/Origin-Quantum/vqnet2-skill.git
cd vqnet2-skill
```

## Requirements for VQNet Development

To use VQNet 2.0 in your own code, you need to install the Python packages:

```bash
# Install VQNet 2.0 and PyQPanda3
pip install pyvqnet
pip install pyqpanda3

# Optional: For PyTorch backend integration
pip install torch>=2.4.0,<2.7.0
```

## Usage

### Automatic Triggering

This skill **automatically triggers** when you mention:
- VQNet, VQNet2, PyVQNet
- QTensor
- Variational quantum circuit / VQC / 变分量子线路
- Quantum neural network / QNN / 量子神经网络
- Quantum machine learning / QML / 量子机器学习
- QLinear, QConv, quantum convolution
- pyqpanda3

### What the Skill Provides

1. **Complete API Reference** - All function signatures from official documentation
2. **Correct Parameter Order** - Avoids common mistakes like `(input, param)` vs `(param, input)`
3. **Backend Awareness** - Knows the differences between `pyvqnet-ad`, `pyvqnet`, and `torch` backends
4. **Working Examples** - Generates complete, runnable code examples
5. **Debugging Help** - Identifies common mistakes like backend mixing, missing `requires_grad`, etc.

### Example Workflow

1. You: "Help me create a 4-qubit variational quantum classifier using VQNet"
2. Skill: Automatically loads the relevant API references
3. Skill: Generates complete working code with correct imports and parameter order
4. You get code that follows official VQNet 2.0 conventions

## Example Usage

### Creating a QTensor

```python
from pyvqnet.tensor import QTensor
from pyvqnet.dtype import *

# Create tensor with gradient tracking
t = QTensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, dtype=kfloat32)

print(t.shape)  # [2, 2]
print(t.ndim)   # 2
```

### Creating a Quantum Layer

```python
from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
from pyvqnet.qnn.pq3.measure import ProbsMeasure
from pyvqnet.tensor import QTensor
import pyqpanda3.core as pq

def quantum_circuit(input, params):
    n_qubits = 4
    machine = pq.CPUQVM()
    qubits = range(n_qubits)
    cir = pq.QCircuit()

    # Encode input data
    for q, x in zip(qubits, input):
        cir << pq.H(q) << pq.RZ(q, x)

    # Variational entanglement
    for i in range(n_qubits - 1):
        cir << pq.CNOT(i, i + 1) << pq.RY(i + 1, params[i])

    prog = pq.QProg()
    prog << cir
    return ProbsMeasure(machine, prog, list(qubits))

# Create layer with 3 variational parameters
layer = QuantumLayer(quantum_circuit, 3)

# Forward pass
data = QTensor([[0.1, 0.2, 0.3, 0.4]])
output = layer(data)
print(output)

# Backward pass
output.backward()
print(layer.m_para.grad)
```

### Switching to PyTorch Backend

```python
import pyvqnet
import torch
from pyvqnet.tensor import QTensor

# Switch to PyTorch backend
pyvqnet.backends.set_backend("torch")

# QTensor now wraps torch.Tensor
t = QTensor([1.0, 2.0, 3.0], requires_grad=True)
print(type(t.data))  # <class 'torch.Tensor'>
```

## Skill Structure

```
vqnet2-skill/
├── SKILL.md                      # Core skill definition for AI assistant
├── CLAUDE.md                     # Assistant guide (this project)
├── README.md                     # This file - installation guide
├── examples/                     # Working example scripts
│   ├── 01-qtensor-basics.py     # QTensor creation and operations
│   ├── 02-quantum-layer.py       # Basic QuantumLayer example
│   ├── 03-qnn-classification.py  # Quantum neural network classification
│   └── 04-pytorch-backend.py     # PyTorch backend integration
└── references/                   # API reference documentation
    ├── qtensor.md               # QTensor class API
    ├── quantum_layers.md        # Quantum layer classes
    ├── quantum_templates.md     # Circuit templates and gates
    ├── measurement.md           # Measurement functions
    ├── classic_nn.md            # Hybrid layers (QLinear, QConv)
    ├── torch_api.md            # PyTorch backend API
    ├── utils.md                # Utility functions
    ├── vqc.md                  # Variational quantum circuits
    └── qnn.md                  # Quantum neural networks
```

## Common Mistakes this Skill Helps Prevent

| Mistake | How the Skill Helps |
|---------|-------------------|
| Wrong parameter order in quantum circuit function (`(param, input)`) | Ensures `(input, param)` which matches VQNet API |
| Incorrect import paths | Always uses the correct module paths like `pyvqnet.qnn.pq3.quantumlayer` |
| Mixing tensors from different backends | Reminds users to switch backend before creating tensors |
| Forgetting `requires_grad=True` for trainable parameters | Ensures gradients are tracked for training |
| Hardcoding QCloud API tokens | Enforces using environment variables |

## Links

- [Official VQNet Documentation](https://vqnet2-tutorial.readthedocs.io/)
- [Origin Quantum Website](https://www.originqc.com.cn/)
- [PyQPanda3 Documentation](https://qcloud.originqc.com.cn/document/qpanda-3/index.html)

## License

Apache License 2.0 - same as VQNet documentation.

## Credits

This skill is based on the official VQNet 2.0 documentation from Origin Quantum.
