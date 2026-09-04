# Quantum Neural Networks (QNN) Reference (v2.18.1)

## Overview

VQNet provides building blocks for quantum neural networks:

- **Quantum layers** that can be inserted into classical neural networks
- **Hybrid quantum-classical training** with automatic differentiation
- **PyTorch integration** for seamless mixed classical-quantum models
- **Multiple gradient methods**: parameter-shift, finite-difference, adjoint

## Hybrid Model Architecture

A typical hybrid quantum-classical QNN looks like:

```
 Classical Input  →  Classical Preprocessing  →  Quantum Embedding  →
 Variational Quantum Circuit  →  Measurement  →  Classical Post-processing →  Output
```

VQNet handles the quantum part with `QuantumLayer` and compatible with classical layers.

## Example: Hybrid QNN for Classification

```python
import pyqpanda3.core as pq
from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
from pyvqnet.qnn.pq3.measure import ProbsMeasure
from pyvqnet.nn import Linear, ReLU, Sequential, CrossEntropyLoss
from pyvqnet.tensor import QTensor
from pyvqnet.optim import Adam

# Define the quantum part
def quantum_forward(x, params):
    n_qubits = 4
    machine = pq.CPUQVM()
    qubits = range(n_qubits)
    cir = pq.QCircuit()

    # Angle encoding
    for i, q in enumerate(qubits):
        if i < len(x):
            cir << pq.H(q) << pq.RY(q, x[i])

    # Entanglement + variational
    for i in range(n_qubits - 1):
        cir << pq.CNOT(i, i + 1)
        cir << pq.RX(i + 1, params[i])

    prog = pq.QProg() << cir
    return ProbsMeasure(machine, prog, qubits)

# Hybrid sequential model
model = Sequential(
    Linear(784, 16),
    ReLU(),
    Linear(16, 4),
    QuantumLayer(quantum_forward, 3),  # Quantum layer
    Linear(4, 10)  # Classical output layer
)

# Training
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = CrossEntropyLoss()

# Forward pass
batch_x = QTensor(...)  # Shape: [batch_size, 784]
batch_y = QTensor(...)  # Shape: [batch_size]
optimizer.zero_grad()
output = model(batch_x)
loss = loss_fn(output, batch_y)
loss.backward()
optimizer.step()
```

## Available QNN Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `QuantumLayer` | `pyvqnet.qnn.pq3.quantumlayer` | Generic variational quantum layer |
| `QpandaQProgVQCLayer` | `pyvqnet.qnn.pq3.quantumlayer` | Layer that accepts QProg return |
| `QuantumBatchAsyncQcloudLayer` | `pyvqnet.qnn.pq3.quantumlayer` | Run on QCloud real hardware |
| `QuantumLayerAdjoint` | `pyvqnet.qnn.pq3.quantumlayer` | Adjoint gradient method for VQE |
| `NoiseQuantumLayer` | `pyvqnet.qnn.pq3.quantumlayer` | Quantum layer with noise model support |
| `QLinear` | `pyvqnet.qnn.qlinear` | Quantum fully-connected layer |
| `QConv` | `pyvqnet.qnn.qcnn.qconv` | Quantum convolution layer |
| Embedding templates | `pyvqnet.qnn.pq3.template` | AmplitudeEmbedding, AngleEmbedding, IQP |
| Ansatz templates | `pyvqnet.qnn.pq3.ansatz` | HardwareEfficientAnsatz |

## Best Practices

### Data Encoding

Choose encoding based on your data dimension:

| Data dimension | Encoding method | Qubits needed |
|----------------|-----------------|---------------|
| N features | Angle embedding | N qubits |
| 2^N features | Amplitude embedding | N qubits |
| Any N ≤ qubits | IQP embedding | N qubits |

### Training

- Use **Adam** optimizer for most cases, SGD with momentum also works
- Start with small learning rate (1e-3 to 1e-4) for variational parameters
- Batch norm can help training like in classical NNs
- Gradient clipping can stabilize training if gradients explode

### Runtime Considerations

- Gradient computation via parameter-shift scales **linearly** with number of parameters
- For many parameters, consider using stochastic coordinate descent (`random_coordinate_descent`)
- CPU simulation is fine for up to ~20 qubits
- Use GPU simulator (`GPUQVM`) for 15-25 qubits
- Use QCloud for larger than 25 qubits or real hardware experiments

## Common Issues

1. **"My circuit doesn't compute gradients"**
   - Check: Is the parameter order correct in your circuit function? It must be `(input, param)`, not `(param, input)`

2. **"Out of memory"**
   - You're simulating too many qubits on CPU. Reduce qubit count or use GPU/QCloud.

3. **"Loss doesn't decrease"**
   - Check: Are your parameters correctly initialized? Are you zeroing gradients each step?
   - Try smaller learning rate. Quantum training can be sensitive to LR.

4. **"Can I mix QTensor from different backends?"**
   - No! Once you switch backend, recreate all tensors. Different backends have different underlying data structures.
