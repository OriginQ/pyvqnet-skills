"""
PyTorch Backend Example
=======================
Demonstrates using VQNet with PyTorch as the computation backend.
"""

import pyvqnet
import torch
from pyvqnet.tensor import QTensor
from pyvqnet.nn import Linear
from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
import pyqpanda3.core as pq
from pyvqnet.qnn.pq3.measure import ProbsMeasure

# Switch to PyTorch backend
pyvqnet.backends.set_backend("torch")
print(f"Current backend: {pyvqnet.backends.get_backend()}")
print()

# Now QTensor uses torch.Tensor internally
def circuit(input, param):
    n_qubits = 4
    machine = pq.CPUQVM()
    qubits = range(n_qubits)
    cir = pq.QCircuit()

    for q in qubits:
        if q < len(input):
            cir << pq.H(q) << pq.RZ(q, input[q])
    for i in range(n_qubits - 1):
        cir << pq.CNOT(i, i + 1)
        cir << pq.RY(i + 1, param[i])

    prog = pq.QProg() << cir
    return ProbsMeasure(machine, prog, qubits)

# Create model - works the same as default backend
model = QuantumLayer(circuit, 3)

# Create input - QTensor wraps torch.Tensor now
x = QTensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.4, 0.3, 0.2]], requires_grad=True)
print(f"Input QTensor created, underlying type: {type(x.data)}")
print()

# Forward pass
y = model(x)
print(f"Output shape: {y.shape}")
print(f"Output:\n{y}")
print()

# Backward pass works through PyTorch autograd
y.sum().backward()
print(f"Gradient computed")
print(f"Parameter gradients:\n{model.m_para.grad}")
