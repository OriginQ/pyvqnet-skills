# Quantum Neural Networks (QNN) Reference (v2.18.2)

## Overview

VQNet 2.18.2 treats a parameterized quantum circuit (PQC) as a differentiable operator and embeds it into classical neural networks for end-to-end hybrid quantum-classical training. It provides two independent compute backends, selected globally with `pyvqnet.backends.set_backend(...)`; a model must stay on one backend.

| Backend | Select with | Key modules |
|---------|-------------|-------------|
| Native (`pyvqnet`) | `pyvqnet.backends.set_backend("pyvqnet")` | `pyvqnet.tensor`, `pyvqnet.nn`, `pyvqnet.qnn.vqc`, `pyvqnet.qnn.vqc.tn` |
| Torch (`torch`) | `pyvqnet.backends.set_backend("torch")` | `pyvqnet.nn.torch`, `pyvqnet.qnn.vqc.sv.torch`, `pyvqnet.qnn.vqc.tn.torch`, `pyvqnet.torch.trl` |

Core capabilities:

- **Native simulation** — `pyvqnet.qnn.vqc` provides the state-vector `QMachine` (50+ gates, `MeasureAll`, `Probability`, `Samples`), `QuantumLayerAdjoint`, quantum-classical layers such as QLSTM/QRNN/QMLP/Quanvolution, and LLM operators.
- **Tensor-network backend** — `pyvqnet.qnn.vqc.tn` provides `native` and `torch` implementations for larger circuits.
- **pyqpanda3 quantum layers** — `pyvqnet.qnn.pq3` includes `QuantumLayer`, `QpandaQProgVQCLayer`, `QuantumLayerAdjoint`, `VQCLayer`, `TorchVQCLayer`, cloud layers, and circuit templates/ansätze.
- **Distributed training** — the `vqnetrun` launcher uses gloo (CPU) or NCCL (GPU) collectives, with data, tensor, pipeline, and hybrid parallelism through `ParallelTrainingWrapper` (`pyvqnet.distributed.hybrid`), plus qubit-parallel `DistributedQMachine` / `DistQuantumLayerAdjoint`.
- **LLM support** — native LLM operators and the `pyvqnet.torch.trl` loss family (SFT, DPO, PPO, GRPO, Reward) for torch-backend fine-tuning.

## Hybrid Model Architecture

A typical hybrid quantum-classical QNN looks like:

```
 Classical Input  →  Classical Preprocessing  →  Quantum Embedding  →
 Variational Quantum Circuit  →  Measurement  →  Classical Post-processing →  Output
```

VQNet handles the quantum part with `QuantumLayer` and composes it with classical layers.

## Example: Hybrid QNN for Classification

```python
import numpy as np
import pyqpanda3.core as pq
from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
from pyvqnet.qnn.pq3.measure import ProbsMeasure
from pyvqnet.nn import Linear, ReLU, Sequential, CrossEntropyLoss
from pyvqnet.tensor import QTensor
from pyvqnet.optim import Adam
from pyvqnet import kint64

# Define the quantum part
def quantum_forward(x, params):  # signature must be (input, param)
    n_qubits = 4
    machine = pq.CPUQVM()
    qubits = range(n_qubits)
    cir = pq.QCircuit()

    # Angle encoding
    for i, q in enumerate(qubits):
        cir << pq.H(q) << pq.RY(q, x[i])

    # Entanglement + variational
    for i in range(n_qubits - 1):
        cir << pq.CNOT(i, i + 1)
        cir << pq.RX(i + 1, params[i])

    prog = pq.QProg()
    prog << cir
    # 2.18.2: measure_qubits must be a list; [0, 1] gives 4 probabilities
    return ProbsMeasure(machine, prog, [0, 1])

# Sequential accepts a list of modules
model = Sequential(
    Linear(784, 16),
    ReLU(),
    Linear(16, 4),
    QuantumLayer(quantum_forward, 3),  # 3 trainable parameters
    Linear(4, 10)                      # classical output layer
)

# Training
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = CrossEntropyLoss()

batch_x = QTensor(np.random.randn(8, 784).astype(np.float32))
batch_y = QTensor(np.random.randint(0, 10, size=(8,)), dtype=kint64)
optimizer.zero_grad()
output = model(batch_x)
loss = loss_fn(batch_y, output)  # VQNet: (y_true, y_pred), labels first
loss.backward()
optimizer._step()
```

## Available QNN Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `QuantumLayer` | `pyvqnet.qnn.pq3.quantumlayer` | Generic variational quantum layer backed by a pyqpanda3 circuit function |
| `QpandaQProgVQCLayer` | `pyvqnet.qnn.pq3.quantumlayer` | Layer that accepts a QProg-returning function |
| `QuantumBatchAsyncQcloudLayer` | `pyvqnet.qnn.pq3.quantumlayer` | Asynchronous batch submission to QCloud |
| `QuantumLayerAdjoint` | `pyvqnet.qnn.pq3.quantumlayer` | Adjoint-gradient VQE layer |
| `VQCLayer` | `pyvqnet.qnn.pq3.vqclayer` | Unified quantum layer for VQC `QMachine` modules; selects `vqnet_native` / `qcloud_fake` / `qpanda_runtime` / `qcloud_service` via `submit_kwargs["backend"]` |
| `TorchVQCLayer` | `pyvqnet.qnn.pq3.torch` | Torch-backend counterpart of `VQCLayer` |
| `QiskitLayer` / `CirqLayer` | `pyvqnet.qnn.utils` | Wrap Qiskit / Cirq circuits as VQNet layers |
| `SPSA` | `pyvqnet.qnn` | Simultaneous perturbation stochastic approximation optimizer |
| `QLinear` | `pyvqnet.qnn.qlinear` | Quantum fully-connected layer |
| `QConv` | `pyvqnet.qnn.qcnn.qconv` | Quantum convolution layer |
| Embedding templates | `pyvqnet.qnn.pq3.template` | AmplitudeEmbeddingCircuit, AngleEmbeddingCircuit, IQPEmbeddingCircuits, BasicEmbeddingCircuit |
| Ansatz templates | `pyvqnet.qnn.pq3.ansatz`, `pyvqnet.qnn.pq3.template` | HardwareEfficientAnsatz, BasicEntanglerTemplate, StronglyEntanglingTemplate, UCCSD |

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
- Start with a small learning rate (1e-3 to 1e-4) for variational parameters
- Batch norm can help training like in classical NNs
- Gradient clipping can stabilize training if gradients explode
- Call `pyvqnet.backends.set_backend(...)` before creating tensors or modules, and do not mix tensors from different backends

### Runtime Considerations

- Gradient computation via parameter-shift scales **linearly** with the number of parameters
- For many parameters, consider `diff_method="random_coordinate_descent"` on `VQCLayer`
- Native state-vector simulation is fine up to ~20 qubits; use the tensor-network backend (`pyvqnet.qnn.vqc.tn`) for larger circuits
- Use GPU simulation or QCloud for larger circuits or real-hardware experiments
- For distributed runs, launch with `vqnetrun`; CPU workers use gloo and GPU workers use NCCL collectives

## Common Issues

1. **"My circuit doesn't compute gradients"**
   - Check the parameter order in the circuit function: it must be `(input, param)`, not `(param, input)`

2. **"ProbsMeasure fails with a scalar qubit index"**
   - Pass a list: `ProbsMeasure(machine, prog, [0, 1])`

3. **"CrossEntropyLoss complains about the label dtype"**
   - Labels must be `kint64`: `QTensor(labels, dtype=kint64)`

4. **"VQC model output is wrong across batches"**
   - Call `self.device.reset_states(batchsize)` at the start of every `forward`

5. **"Out of memory"**
   - You are simulating too many qubits on CPU. Reduce the qubit count or use the tensor-network/GPU/QCloud path

6. **"Can I mix QTensor from different backends?"**
   - No. Once you switch backend, recreate all tensors and modules. Different backends have different underlying data structures
