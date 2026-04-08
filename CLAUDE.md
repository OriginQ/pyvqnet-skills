# CLAUDE.md - VQNet 2.0 API Skill

## Project Overview

**Purpose**: AI skill package for helping users develop quantum machine learning models using VQNet 2.0 with pyqpanda3. Provides complete API reference extracted from official documentation and helps with code generation, debugging, and best practices.

**Primary Framework**: VQNet 2.0 (PyVQNet) with pyqpanda3 for quantum computing.

**Target Users**: AI assistants (Claude/Cline) helping users with quantum machine learning using VQNet 2.0.

**License**: Apache 2.0

## When to Use This Skill

Use this skill when:
- User asks about VQNet API or QTensor
- User needs to create a quantum neural network
- User wants to implement variational quantum circuits (VQC)
- User needs help with PyTorch backend integration
- User asks about quantum convolution or quantum fully-connected layers
- User needs help debugging VQNet code

## Key Files to Reference

| Topic | Reference File |
|-------|----------------|
| QTensor tensor class | `references/qtensor.md` |
| Quantum layers (QuantumLayer, etc.) | `references/quantum_layers.md` |
| Circuit templates & gate functions | `references/quantum_templates.md` |
| Measurement functions | `references/measurement.md` |
| Classical/quantum hybrid NN layers (QLinear, QConv) | `references/classic_nn.md` |
| PyTorch backend integration | `references/torch_api.md` |
| Utility functions | `references/utils.md` |
| Variational quantum circuits | `references/vqc.md` |
| Quantum neural networks | `references/qnn.md` |

## Architecture Overview

```
vqnet2-skill/
├── SKILL.md                      # Core skill definition with workflow
├── CLAUDE.md                     # This file - AI assistant guide
├── README.md                     # Human-readable README
├── examples/                     # Executable example scripts
│   ├── 01-qtensor-basics.py     # QTensor creation and operations
│   ├── 02-quantum-layer.py       # Basic QuantumLayer example
│   ├── 03-qnn-classification.py  # Quantum neural network classification
│   └── 04-pytorch-backend.py     # PyTorch backend integration example
└── references/                   # API reference documentation
    ├── qtensor.md               # QTensor API
    ├── quantum_layers.md        # Quantum layer classes
    ├── quantum_templates.md     # Circuit templates and gates
    ├── measurement.md           # Measurement and entropy functions
    ├── classic_nn.md            # QLinear, QConv
    ├── torch_api.md            # PyTorch backend
    ├── utils.md                # Utility functions
    ├── vqc.md                  # Variational quantum circuits
    └── qnn.md                  # Quantum neural networks
```

---

## Critical API Patterns to Remember

### QTensor Key Points

```python
from pyvqnet.tensor import QTensor
from pyvqnet.dtype import *

# Constructor: QTensor(data, requires_grad=False, device=DEV_CPU, dtype=None)
t = QTensor([1.0, 2.0, 3.0], requires_grad=True)

# Common attributes/methods:
t.shape        # list of dimensions
t.ndim         # number of dimensions
t.size         # number of elements
t.dtype        # data type (kfloat32, kfloat64, etc.)
t.requires_grad # whether gradients are tracked
t.grad         # gradient after backward()
t.zero_grad()  # zero out gradients
t.backward()   # compute gradients via backprop
t.to_numpy()   # convert to numpy array
t.numel()      # number of elements (same as size)
```

### QuantumLayer Signatures

**QuantumLayer** (most common):
```python
QuantumLayer(qprog_with_measure, para_num, diff_method="parameter_shift",
             delta=0.01, dtype=None, name="")
```
- `qprog_with_measure` must be a function `def func(input, param): ...`
- Function signature: **input first, then param** (this is easy to get wrong)
- Function must return measurement result (np.ndarray or list)

**QpandaQProgVQCLayer** (alias: QuantumLayerV3):
```python
QpandaQProgVQCLayer(origin_qprog_func, para_num, qvm_type="cpu",
                    pauli_str_dict=None, shots=1000,
                    initializer=None, dtype=None, name="")
```
- `origin_qprog_func` must return `pyqpanda3.core.QProg`
- Same function signature requirement: `(input, param)`

**QuantumBatchAsyncQcloudLayer**:
For running on Origin Quantum QCloud real hardware. Requires API token.

**QuantumLayerAdjoint**:
Uses adjoint method for gradients with pyqpanda3 VQCircuit.

### Measurement Functions

```python
from pyvqnet.qnn.pq3.measure import expval, ProbsMeasure, QuantumMeasure

# Expectation value of Pauli string
expval(machine, prog, pauli_str_dict)  # -> float

# Probability measurement
ProbsMeasure(machine, prog, measure_qubits, shots=1)  # -> np.array

# Full quantum measurement with shots
QuantumMeasure(machine, prog, measure_qubits, shots=1000)  # -> dict
```

---

## Common Mistakes to Watch For

1. **Wrong parameter order** in quantum circuit function:
   - CORRECT: `def circuit(input, param): ...`
   - WRONG: `def circuit(param, input): ...`

2. **Mixing backends**: QTensor from different backends can't be mixed. If user switches backend, they need to recreate tensors.

3. **Forgetting requires_grad**: Trainable parameters must have `requires_grad=True`.

4. **Wrong import paths**: Make sure to use correct module paths:
   - `pyvqnet.qnn.pq3.quantumlayer` for quantum layers
   - `pyvqnet.qnn.pq3.measure` for measurement
   - `pyvqnet.qnn.pq3.template` for templates
   - `pyvqnet.tensor` for QTensor

5. **QCloud token**: Never hardcode the token in example code - use `os.getenv("QCLOUD_TOKEN")`.

---

## Dependencies

Users need to install:
```bash
pip install pyvqnet
pip install pyqpanda3
# Optional for PyTorch backend:
pip install torch>=2.4.0,<2.7.0
```

---

## Verification Checklist

When generating VQNet code:

- [ ] Correct import paths for all modules
- [ ] Quantum circuit function has correct signature `(input, param)`
- [ ] QTensor created with correct `requires_grad` setting
- [ ] Backend selection is clear if using non-default
- [ ] Example includes complete imports and can run standalone
- [ ] No hardcoded QCloud API tokens
- [ ] Parameter order matches API documentation

---

## Additional Resources

- [Official VQNet Documentation](https://vqnet2-tutorial.readthedocs.io/)
- [Origin Quantum](https://www.originqc.com.cn/)
- [PyQPanda3 Documentation](https://qcloud.originqc.com.cn/document/qpanda-3/index.html)
