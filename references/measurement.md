# Measurement Functions API Reference

## expval

```python
pyvqnet.qnn.pq3.measure.expval(machine, prog, pauli_str_dict)
```

**Description:**
Computes the expectation value of a Hamiltonian given as a dictionary of Pauli strings.

For a Hamiltonian `0.7 Z0⊗X1⊗I2 + 0.2 I0⊗Z1⊗I2`, the dictionary looks like:
`{'Z0 X1': 0.7, 'Z1': 0.2}`

Supports pyqpanda3 CPU and GPU simulators.

**Parameters:**
- `machine` - Quantum virtual machine created by pyqpanda
- `prog` - Quantum program created by pyqpanda
- `pauli_str_dict` - Dictionary mapping Pauli strings to coefficients
- **Returns:** Expectation value (float)

**Example:**
```python
import pyqpanda3.core as pq
from pyvqnet.qnn.pq3.measure import expval

input = [0.56, 0.1]
m_machine = pq.CPUQVM()
m_qlist = range(3)
cir = pq.QCircuit(3)
cir << pq.RZ(m_qlist[0], input[0])
cir << pq.CNOT(m_qlist[0], m_qlist[1])
cir << pq.RY(m_qlist[1], input[1])
cir << pq.CNOT(m_qlist[0], m_qlist[2])
m_prog = pq.QProg(cir)

pauli_dict  = {'Z0 X1': 10, 'Y2': -0.543}
exp2 = expval(m_machine, m_prog, pauli_dict)
print(exp2)
```

---

## QuantumMeasure

```python
pyvqnet.qnn.pq3.measure.QuantumMeasure(
    machine,
    prog,
    measure_qubits: list,
    shots: int = 1000,
    qcloud_option=""
)
```

**Description:**
Performs quantum measurement using Monte Carlo method. Returns measurement outcome counts.

Only supports `CPUQVM` and `QCloud`.

**Parameters:**
- `machine` - Quantum VM allocated by pyQPanda
- `prog` - Quantum program created by pyQPanda
- `measure_qubits` - `list[int]` - List of measured qubit indices
- `shots` - `int` - Number of measurement shots, default: 1000
- `qcloud_option` - `QCloudOptions` - QCloud configuration, only used with QCloud, default: ""
- **Returns:** Measurement results (dictionary with outcome counts)

**Example:**
```python
from pyqpanda3.core import *
from pyvqnet.qnn.pq3.measure import QuantumMeasure

circuit = QCircuit(3)
circuit << H(0) << RX(1, 0.9) << RX(0, 0.6) << RX(1, 0.3)
circuit << RY(1, 0.3) << RY(2, 2.7) << RX(0, 1.5)
prog = QProg()
prog.append(circuit)
machine = CPUQVM()

measure_result = QuantumMeasure(machine, prog, [2, 0], shots=1000)
print(measure_result)
```

---

## ProbsMeasure

```python
pyvqnet.qnn.pq3.measure.ProbsMeasure(
    machine,
    prog,
    measure_qubits: list,
    shots=1
)
```

**Description:**
Computes probability measurement. For `shots=1` (default), computes the theoretical probability distribution.

Only supports `CPUQVM` and `QCloud`.

**Parameters:**
- `machine` - pyQPanda quantum VM
- `prog` - Quantum program created by pyQPanda
- `measure_qubits` - `list[int]` - List of measured qubit indices
- `shots` - `int` - Number of measurement shots, default: 1 (theoretical calculation)
- **Returns:** Probability array in lex order of measured qubits

**Example:**
```python
from pyqpanda3.core import *
from pyvqnet.qnn.pq3.measure import ProbsMeasure

circuit = QCircuit(3)
circuit << H(0) << RX(1, 0.9) << RX(0, 0.6) << RX(1, 0.3)
circuit << RY(1, 0.3) << RY(2, 2.7) << RX(0, 1.5)
prog = QProg()
prog.append(circuit)
prog.append(measure(0, 0))
prog.append(measure(1, 1))
prog.append(measure(2, 2))

machine = CPUQVM()
measure_result = ProbsMeasure(machine, prog, [2, 0], shots=1000)
print(measure_result)
# [0.0479..., 0, 0.476..., 0.476...]
```

---

## DensityMatrixFromQstate

```python
pyvqnet.qnn.pq3.measure.DensityMatrixFromQstate(state, indices)
```

**Description:**
Computes the reduced density matrix for a subset of qubits given the full quantum state vector.

**Parameters:**
- `state` - `list[complex]` - Full state vector from `get_qstate()`. Size must be `2^N` for N qubits. Qubit order is `000 -> 111` (lex order).
- `indices` - `list[int]` - Indices of qubits for the reduced density matrix
- **Returns:** `np.ndarray` - Reduced density matrix of shape `[2^len(indices), 2^len(indices)]`

**Example:**
```python
from pyvqnet.qnn.pq3.measure import DensityMatrixFromQstate

qstate = [
    (0.9306699299765968+0j), (0.18865613455240968+0j),
    (0.1886561345524097+0j), (0.03824249173404786+0j),
    -0.048171819846746615j, -0.00976491131165138j,
    -0.23763904794287155j, -0.048171819846746615j
]
print(DensityMatrixFromQstate(qstate, [0, 1]))
# [[0.868...  0.187...  0.176...  0.037...]
#  [0.187...  0.092...  0.037...  0.018...]
#  [0.176...  0.037...  0.035...  0.007...]
#  [0.037...  0.018...  0.007...  0.003...]]
```

---

## VN_Entropy

```python
pyvqnet.qnn.pq3.measure.VN_Entropy(state, indices, base=None)
```

**Description:**
Computes Von Neumann entropy S(ρ) = -Tr(ρ log ρ) for a subset of qubits given the full state.

**Formula:** S(ρ) = -Tr(ρ log ρ)

**Parameters:**
- `state` - `list[complex]` - Full state vector from `get_qstate()`. Size `2^N` for N qubits, lex order 000 -> 111.
- `indices` - `list[int]` - Qubit indices to compute entropy for
- `base` - Logarithm base. If None, uses natural log (base e), default: None
- **Returns:** `float` - Von Neumann entropy value

**Example:**
```python
from pyvqnet.qnn.pq3.measure import VN_Entropy

qstate = [
    (0.9022961387408862 + 0j), -0.06676534788028633j,
    (0.18290448232350312 + 0j), -0.3293638014158896j,
    (0.03707657410649268 + 0j), -0.06676534788028635j,
    (0.18290448232350312 + 0j), -0.013534006039561714j
]
print(VN_Entropy(qstate, [0, 1]))
# 0.14592917648464448
```

---

## Mutal_Info

```python
pyvqnet.qnn.pq3.measure.Mutal_Info(
    state,
    indices0,
    indices1,
    base=None
)
```

**Description:**
Computes mutual information I(A, B) = S(ρ^A) + S(ρ^B) - S(ρ^{AB}) between two subsystems.

**Parameters:**
- `state` - `list[complex]` - Full state vector from `get_qstate()`
- `indices0` - `list[int]` - Qubit indices for first subsystem
- `indices1` - `list[int]` - Qubit indices for second subsystem
- `base` - Logarithm base. If None, uses natural log, default: None
- **Returns:** `float` - Mutual information between the two subsystems

**Example:**
```python
from pyvqnet.qnn.pq3.measure import Mutal_Info

qstate = [
    (0.9022961387408862 + 0j), -0.06676534788028633j,
    (0.18290448232350312 + 0j), -0.3293638014158896j,
    (0.03707657410649268 + 0j), -0.06676534788028635j,
    (0.18290448232350312 + 0j), -0.013534006039561714j
]
print(Mutal_Info(qstate, [0], [2], 2))
# 0.13763425302805887
```

---

## Purity

```python
pyvqnet.qnn.pq3.measure.Purity(state, qubits_idx)
```

**Description:**
Computes the purity γ = Tr(ρ²) for specific qubits from the full state.

For normalized quantum states: 1/d ≤ γ ≤ 1 where d is the Hilbert space dimension. Pure states have purity 1.

**Formula:** γ = Tr(ρ²)

**Parameters:**
- `state` - Full quantum state vector from `pyqpanda` `get_qstate()`
- `qubits_idx` - `list[int]` - Qubit indices to compute purity for
- **Returns:** `float` - Purity value

**Example:**
```python
from pyvqnet.qnn.pq3.measure import Purity

qstate = [
    (0.9306699299765968 + 0j), (0.18865613455240968 + 0j),
    (0.1886561345524097 + 0j), (0.03824249173404786 + 0j),
    -0.048171819846746615j, -0.00976491131165138j,
    -0.23763904794287155j, -0.048171819846746615j
]
pp = Purity(qstate, [1])
print(pp)
# 0.902503479761881
```
