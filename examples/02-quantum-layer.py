"""
QuantumLayer Example
====================
Demonstrates creating a simple variational quantum layer with VQNet.
"""

from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
from pyvqnet.qnn.pq3.measure import ProbsMeasure
from pyvqnet.tensor import QTensor, ones
from pyvqnet.optim import SGD
import pyqpanda3.core as pq

def simple_vqc(input, param):
    """ A simple variational quantum circuit.

    Args:
        input: 1D input data (4 features)
        param: 3 variational parameters

    Returns:
        Probability measurement of qubits
    """
    num_qubits = 4
    machine = pq.CPUQVM()
    qubits = range(num_qubits)
    circuit = pq.QCircuit()

    # Hadamard + encode input with RZ rotation
    for q in qubits:
        circuit << pq.H(q)
        if q < len(input):
            circuit << pq.RZ(q, input[q])

    # Entangling layers with variational parameters
    for i in range(num_qubits - 1):
        circuit << pq.CNOT(i, i + 1)
        if i < len(param):
            circuit << pq.RY(i + 1, param[i])

    prog = pq.QProg()
    prog << circuit
    return ProbsMeasure(machine, prog, list(qubits))

# Create quantum layer with 3 parameters
layer = QuantumLayer(simple_vqc, 3)

# Input data: batch of 3 samples, 4 features each
input_data = QTensor([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.3, 0.2, 0.1],
    [1.0, 0.0, 0.5, 0.5]
])

# Forward pass
print("Forward pass:")
output = layer(input_data)
print(output)
print()

# Backward pass
print("Backward pass:")
grad = ones(output.shape)
output.backward(grad)
print(f"Parameter gradients:\n{layer.m_para.grad}")
