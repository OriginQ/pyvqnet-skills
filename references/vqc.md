# Variational Quantum Circuits (VQC) Reference

## Overview

Variational Quantum Circuits (VQC) are parameterized quantum circuits used in variational quantum algorithms (VQA) including:
- Variational Quantum Eigensolver (VQE) for quantum chemistry
- Quantum Approximate Optimization Algorithm (QAOA) for combinatorial optimization
- Variational quantum classifiers for machine learning
- Hybrid quantum-classical neural networks

## Key Concepts

In VQNet/pyqpanda3:
1. **Users define the circuit function** with signature `def circuit(input, param): ...`
2. **Input**: classical data to encode (1D array)
3. **Param**: trainable variational parameters (1D array)
4. **Function returns**: either measurement result (for QuantumLayer) or QProg (for QpandaQProgVQCLayer)
5. **VQNet handles autograd** using parameter-shift or adjoint method

## Common Patterns

### Basic VQC Classification

```python
import pyqpanda3.core as pq
from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
from pyvqnet.qnn.pq3.measure import ProbsMeasure
from pyvqnet.tensor import QTensor
from pyvqnet.nn import CrossEntropyLoss
from pyvqnet.optim import SGD

# Define the variational quantum circuit
def classifier_circuit(input, params):
    n_qubits = 4
    machine = pq.CPUQVM()
    qubits = range(n_qubits)
    cir = pq.QCircuit()

    # Hadamard + encode input
    for q in qubits:
        cir << pq.H(q) << pq.RZ(q, input[q])

    # Variational layers with entanglement
    for i in range(n_qubits - 1):
        cir << pq.CNOT(i, i + 1)
        cir << pq.RY(i + 1, params[i])

    prog = pq.QProg()
    prog << cir
    return ProbsMeasure(machine, prog, list(qubits))

# Create model: 4 input features -> 2 output probabilities
model = QuantumLayer(classifier_circuit, 3)

# Optimizer
optimizer = SGD(model.parameters(), lr=0.01)
loss_fn = CrossEntropyLoss()

# Training step
data = QTensor([[0.1, 0.2, 0.3, 0.4]])  # Batch size 1
label = QTensor([0], dtype=kint64)

optimizer.zero_grad()
output = model(data)
loss = loss_fn(output, label)
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")
```

## Gradient Methods Comparison

| Method | Description | Use Case |
|--------|-------------|----------|
| Parameter-shift | General-purpose, works for any differentiable gate | Most cases, any circuit |
| Finite difference | Simple, numerically less stable | Testing, debugging |
| Adjoint method | More efficient for VQE | Hamiltonian expectation, uses VQCircuit |

## Function Signature Requirements

**CRITICAL: This is a common source of errors:**

For `QuantumLayer`:
```python
# CORRECT
def my_circuit(input, params):
    # ... build circuit ...
    return measurement_result  # np.ndarray or list

# WRONG - swapped parameters
def my_circuit(params, input):  # Don't do this!
    ...
```

For `QpandaQProgVQCLayer`:
```python
# CORRECT
def my_circuit(input, params):
    # ... build circuit ...
    return qprog  # pyqpanda3.core.QProg
```

## Best Practices

1. **Keep circuits shallow** - Shallow circuits are faster to simulate and less prone to error
2. **Use CPUQVM for development** - Only use QCloud/hardware when you're ready for production
3. **Check parameter count** - Make sure `para_num` matches the actual number of parameters
4. **Normalize data** - For amplitude embedding, L2 norm must be 1
5. **Set requires_grad** - Don't forget for trainable parameters
