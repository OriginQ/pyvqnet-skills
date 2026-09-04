"""
PyTorch Backend Example
=======================
Demonstrates using VQNet with PyTorch as the computation backend.

注意: torch 后端下的量子线路需使用专用模块 `pyvqnet.qnn.vqc.sv.torch`（态矢）
或 `pyvqnet.qnn.vqc.tn.torch`（张量网络）。pq3 的 `QuantumLayer`
（pyqpanda3 模拟器层）在 torch 后端下不支持 forward。
"""

import pyvqnet
from pyvqnet.tensor import QTensor

# Switch to PyTorch backend
pyvqnet.backends.set_backend("torch")
print(f"Current backend: {pyvqnet.backends.get_backend()}")
print()

# QTensor wraps torch.Tensor now
x = QTensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.4, 0.3, 0.2]], requires_grad=True)
print(f"Input QTensor created, underlying type: {type(x.data)}")
print()

# Quantum circuit under torch backend (see references/torch_api.md)
from pyvqnet.qnn.vqc.sv.torch import QMachine, RX, RZ, Probability, rx, rz, cnot

batchsize = x.shape[0]
device = QMachine(4)

# Key: must reset states at the start of every forward pass
device.reset_states(batchsize)

# Angle encoding via function-form gates (params come from torch-backed QTensor)
rz(q_machine=device, wires=0, params=x[:, [0]])
rz(q_machine=device, wires=1, params=x[:, [1]])
rx(q_machine=device, wires=2, params=x[:, [2]])
rx(q_machine=device, wires=3, params=x[:, [3]])

# Entanglement
cnot(q_machine=device, wires=[0, 1])
cnot(q_machine=device, wires=[2, 3])

# Trainable gates (parameters are torch.nn.Parameter under the hood)
rx_layer = RX(has_params=True, trainable=True, wires=0)
rz_layer = RZ(has_params=True, trainable=True, wires=1)
rx_layer(q_machine=device)
rz_layer(q_machine=device)

# Measurement
measure = Probability(wires=[0, 1])
y = measure(q_machine=device)
print(f"Output shape: {y.shape}")
print(f"Output:\n{y}")
print()

# Backward pass works through PyTorch autograd
y.sum().backward()
print("Gradient computed")
print(f"Trainable RX parameter grad:\n{rx_layer.params.grad}")
print(f"Input gradients:\n{x.grad}")
