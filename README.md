# VQNet 2.0 AI Skill

AI assistant skill for VQNet 2.0/PyVQNet quantum machine learning development.

## Overview

This skill provides AI assistants (Claude, Cline, etc.) with complete API reference for VQNet 2.0, extracted from the official reStructuredText documentation. It helps with:

- **API lookup** - Quick access to function signatures, parameters, and examples
- **Code generation** - Generate correct VQNet code following official patterns
- **Debugging** - Identify common mistakes (wrong parameter order, dtype errors, etc.)
- **Best practices** - Follow the official documentation conventions

## Supported APIs

| Category | Key APIs |
|----------|----------|
| **QTensor** | Core tensor with automatic differentiation, GPU support |
| **Classical NN** | Module, Linear, Conv2D, BatchNorm, LSTM, Loss, Optimizer |
| **QuantumLayer (pyqpanda3)** | QuantumLayer, QuantumBatchAsyncQcloudLayer, QuantumLayerAdjoint |
| **VQC Autograd** | QMachine, Hadamard/RX/RY/RZ/CNOT, Probability, reset_states |
| **Hybrid Layers** | QLinear (quantum FC), QConv (quantum convolution) |
| **Templates** | HardwareEfficientAnsatz, AmplitudeEmbedding, AngleEmbedding |
| **Distributed** | MPI/NCCL multi-GPU training (Linux only) |
| **Quantum LLM** | Fine-tuning with quantum circuits via quantum-llm |

## Installation

### For Claude Code (Claude CLI)

```bash
# Clone the repository
git clone https://gitlab.qpanda.cn/qml/pyvqnet-skills.git

# Copy to Claude Code skills directory
cp -r pyvqnet-skills ~/.claude/skills/vqnet2-api
```

### For VS Code Cline / Roo Code

1. Open Cline settings
2. Find "Custom Instructions" or "Skill Packages" section
3. Add the path to this repository

### Requirements for VQNet Development

```bash
pip install pyvqnet
pip install pyqpanda3

# Optional:
pip install torch>=2.4.0,<2.7.0     # PyTorch backend
conda install conda-forge::mpich-mpicxx==4.1.2  # Distributed CPU
pip install mpi4py                  # Distributed CPU
```

## Usage

### Automatic Triggering

This skill triggers when you mention:
- VQNet, pyvqnet, QTensor
- QuantumLayer, VQC, pyqpanda3
- Quantum neural networks, variational quantum circuits
- Quantum machine learning, QVC, VSQL, Quanvolution

### Example: QuantumLayer

```python
from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
from pyvqnet.qnn.pq3.measure import ProbsMeasure
import pyqpanda3.core as pq

def circuit(input, param):  # 注意：input 在前，param 在后
    machine = pq.CPUQVM()
    qubits = range(4)
    cir = pq.QCircuit()

    for q, x in zip(qubits, input):
        cir << pq.H(q) << pq.RZ(q, x)

    for i in range(3):
        cir << pq.CNOT(i, i + 1) << pq.RY(i + 1, param[i])

    prog = pq.QProg()
    prog << cir
    return ProbsMeasure(machine, prog, list(qubits))

layer = QuantumLayer(circuit, 3)  # 3 个可训练参数
```

### Example: VQC Autograd Module

```python
from pyvqnet.qnn.vqc import QMachine, RZ, Probability
from pyvqnet.nn import Module, Linear

class QModel(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(4, 2)
        self.encode = RZ(wires=0)
        self.device = QMachine(4)

    def forward(self, x):
        # 必须在 forward 开头调用！
        self.device.reset_states(x.shape[0])
        y = self.linear(x)
        self.encode(params=y[:, 0], q_machine=self.device)
        return Probability(wires=[0])(q_machine=self.device)
```

### Example: Complete Training

```python
from pyvqnet.nn import Module, Linear, CrossEntropyLoss
from pyvqnet.optim import Adam
from pyvqnet.tensor import QTensor
from pyvqnet import kint64

model = Linear(10, 5)
optimizer = Adam(model.parameters(), lr=0.01)
loss_fn = CrossEntropyLoss()

x = QTensor([[0.1, ...]], requires_grad=True)
y = QTensor([0], dtype=kint64)  # 标签必须是 kint64

pred = model(x)
loss = loss_fn(y, pred)  # 注意：VQNet 是 (标签, 预测值)

optimizer.zero_grad()
loss.backward()
optimizer._step()  # 注意：是 _step() 而非 step()
```

## Critical API Patterns

| Pattern | VQNet | PyTorch (对比) |
|---------|--------|----------------|
| QuantumLayer 签名 | `(input, param)` | - |
| 损失函数参数 | `(y_true, y_pred)` | `(y_pred, y_true)` |
| CrossEntropy 标签 dtype | `kint64` | `torch.long` |
| 优化器更新 | `optimizer._step()` | `optimizer.step()` |
| VQC forward | 必须调用 `reset_states(batchsize)` | - |

## Project Structure

```
vqnet2-skill/
├── SKILL.md                      # Core skill - 12-layer skill system
├── CLAUDE.md                     # AI assistant guide
├── README.md                     # This file
├── examples/                     # Working example scripts
│   ├── 01-qtensor-basics.py
│   ├── 02-quantum-layer.py
│   ├── 03-qnn-classification.py
│   ├── 04-pytorch-backend.py
│   ├── 05-vqc-autograd.py        # VQC reset_states 示例
│   ├── 06-gpu-training.py        # GPU toGPU() 示例
│   └── 07-complete-training.py   # 完整训练循环
└── references/                   # API reference (from official RST)
    ├── install_env.md            # 安装 + FAQ
    ├── qtensor.md                # QTensor API
    ├── classic_nn.md             # Module, Linear, Conv, Loss, Optimizer
    ├── quantum_layers.md         # QuantumLayer, QcloudLayer
    ├── vqc.md                    # VQC autograd module
    ├── qml_demos.md              # QVC, QDRL, Quanvolution
    ├── distributed.md            # MPI/NCCL distributed
    └── quantum_llm.md            # Quantum LLM fine-tuning
```

## Common Mistakes Prevented

| Mistake | Fix |
|---------|-----|
| `def circuit(param, input)` | Use `(input, param)` |
| `loss_fn(pred, y)` | Use `loss_fn(y, pred)` |
| Labels as `kfloat32` | Use `kint64` for CrossEntropy |
| `optimizer.step()` | Use `optimizer._step()` |
| Missing `reset_states` | Call `device.reset_states(batchsize)` in forward |
| Python `list` for submodules | Use `ModuleList` |
| Hardcoded QCloud token | Use `os.getenv("QCLOUD_TOKEN")` |

## Links

- [Official VQNet Documentation](https://vqnet2-tutorial.readthedocs.io/)
- [Origin Quantum](https://www.originqc.com.cn/)
- [PyQPanda3 Documentation](https://qcloud.originqc.com.cn/document/qpanda-3/index.html)
- [Origin Quantum Cloud](https://qcloud.originqc.com.cn/)

## License

Apache License 2.0

## Credits

Based on official VQNet 2.0 documentation from Origin Quantum.