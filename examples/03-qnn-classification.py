"""
Quantum Neural Network Classification Example
============================================
Demonstrates a hybrid classical-quantum neural network for classification.
"""

import pyqpanda3.core as pq
from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
from pyvqnet.qnn.pq3.measure import ProbsMeasure
from pyvqnet.nn import Linear, ReLU, Sequential, CrossEntropyLoss
from pyvqnet.tensor import QTensor
from pyvqnet import kint64
from pyvqnet.optim import Adam
from pyvqnet.utils import set_random_seed

# Set random seed for reproducibility
set_random_seed(42)

# Define the quantum embedding circuit
def quantum_circuit(input_data, params):
    """
    4 input features -> 4 output probabilities
    """
    n_qubits = 4
    machine = pq.CPUQVM()
    qubits = range(n_qubits)
    cir = pq.QCircuit()

    # Angle encoding of input features
    for i, q in enumerate(qubits):
        if i < len(input_data):
            cir << pq.H(q)
            cir << pq.RY(q, input_data[i])

    # Variational layers with entanglement
    for layer in range(2):
        # Entanglement
        for i in range(n_qubits - 1):
            cir << pq.CNOT(i, i + 1)
        # Rotations
        for i, q in enumerate(qubits):
            idx = layer * n_qubits + i
            if idx < len(params):
                cir << pq.RX(q, params[idx])

    prog = pq.QProg()
    prog << cir
    return ProbsMeasure(machine, prog, list(qubits))

# Create hybrid model: classical -> quantum -> classical
model = Sequential(
    Linear(8, 4),        # Classical: 8 -> 4
    ReLU(),
    QuantumLayer(quantum_circuit, 8),  # Quantum: 4 qubits -> 2^4 = 16 probabilities
    Linear(16, 2)        # Output: 2 classes
)

# Setup optimizer and loss
optimizer = Adam(model.parameters(), lr=0.01)
loss_fn = CrossEntropyLoss()

# Create sample batch (batch size 2, 8 features)
batch_x = QTensor([
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
], requires_grad=True)
batch_y = QTensor([0, 1], dtype=kint64)

# Training step
print("=== Training Step ===")
print(f"Input shape: {batch_x.shape}")

optimizer.zero_grad()
output = model(batch_x)
loss = loss_fn(batch_y, output)  # VQNet: (label, prediction)
loss.backward()
optimizer._step()  # _step() 与 step() 等价

print(f"Output logits:\n{output}")
print()
print(f"Loss: {loss.item():.4f}")
