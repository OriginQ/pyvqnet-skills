# Quantum Circuit Templates and Gates API Reference

## Embedding Circuits

### AmplitudeEmbeddingCircuit

```python
pyvqnet.qnn.pq3.template.AmplitudeEmbeddingCircuit(input_feat, qubits)
```

**Description:**
Encodes `2^n` features into the amplitude of `n` qubits. The L2 norm of features must be 1 for a valid quantum state.

**Parameters:**
- `input_feat` - `np.ndarray` - Feature array
- `qubits` - List[int] - Quantum qubit indices
- **Returns:** `pyqpanda3.QCircuit`

**Example:**
```python
import numpy as np
import pyqpanda3.core as pq
from pyvqnet.qnn.pq3.template import AmplitudeEmbeddingCircuit

input_feat = np.array([2.2, 1, 4.5, 3.7])
input_feat = input_feat / np.linalg.norm(input_feat)  # Normalize!
qlist = range(2)
m_prog = pq.QProg()
cir = AmplitudeEmbeddingCircuit(input_feat, qlist)
m_prog << cir
```

---

### BasicEmbeddingCircuit

```python
pyvqnet.qnn.pq3.template.BasicEmbeddingCircuit(input_feat, qlist)
```

**Description:**
Encodes n binary features into the basis states of n qubits. For `features = [0, 1, 1]`, the quantum state becomes `|011⟩`.

**Parameters:**
- `input_feat` - `(n)` shape - Binary input features (0 or 1)
- `qlist` - List[int] - Qubit indices for the circuit
- **Returns:** `pyqpanda3.QCircuit`

**Example:**
```python
from pyvqnet.qnn.pq3.template import BasicEmbeddingCircuit
import pyqpanda3.core as pq
from pyvqnet import tensor

input_feat = tensor.QTensor([1, 1, 0])
qlist = range(3)
circuit = BasicEmbeddingCircuit(input_feat, qlist)
print(circuit)
```

---

### AngleEmbeddingCircuit

```python
pyvqnet.qnn.pq3.template.AngleEmbeddingCircuit(
    input_feat,
    qubits,
    rotation: str = 'X'
)
```

**Description:**
Encodes N features into rotation angles on n qubits, where `N ≤ n`. Each feature is encoded as a rotation angle around the specified axis.

**Parameters:**
- `input_feat` - `np.ndarray` - Feature array
- `qubits` - List[int] - Qubit indices
- `rotation` - `str` - Rotation axis: 'X', 'Y', or 'Z', default: 'X'
- **Returns:** `pyqpanda3.QCircuit`

**Notes:**
If the number of features is less than the number of qubits, the remaining qubits don't get rotation gates.

**Example:**
```python
from pyvqnet.qnn.pq3.template import AngleEmbeddingCircuit
import numpy as np

m_qlist = range(2)
input_feat = np.array([2.2, 1])
C = AngleEmbeddingCircuit(input_feat, m_qlist, 'X')
print(C)
C = AngleEmbeddingCircuit(input_feat, m_qlist, 'Y')
print(C)
C = AngleEmbeddingCircuit(input_feat, m_qlist, 'Z')
print(C)
```

---

### IQPEmbeddingCircuits

```python
pyvqnet.qnn.pq3.template.IQPEmbeddingCircuits(
    input_feat,
    qubits,
    rep: int = 1
)
```

**Description:**
Instantaneous Quantum Polynomial (IQP) embedding as proposed by Havlicek et al. (2018). Encodes n features into n qubits using repeated IQP blocks.

**Parameters:**
- `input_feat` - `np.ndarray` - Feature array
- `qubits` - List[int] - Qubit indices
- `rep` - `int` - Number of repetitions of the basic IQP block, default: 1
- **Returns:** `pyqpanda3.QCircuit`

**Reference:** https://arxiv.org/pdf/1804.11326.pdf

**Example:**
```python
import numpy as np
from pyvqnet.qnn.pq3.template import IQPEmbeddingCircuits

input_feat = np.arange(1, 100)
qlist = range(3)
circuit = IQPEmbeddingCircuits(input_feat, qlist, rep=3)
print(circuit)
```

---

## Single-Qubit Rotations

### RotCircuit

```python
pyvqnet.qnn.pq3.template.RotCircuit(para, qubits)
```

**Description:**
Arbitrary single-qubit rotation: `R(φ, θ, ω) = RZ(ω) RY(θ) RZ(φ)`

**Matrix:**
```
[ e^{-i(φ+ω)/2} cos(θ/2)   -e^{i(φ-ω)/2} sin(θ/2) ]
[ e^{-i(φ-ω)/2} sin(θ/2)    e^{i(φ+ω)/2} cos(θ/2) ]
```

**Parameters:**
- `para` - `np.ndarray` - `[φ, θ, ω]` parameters
- `qubits` - `int` - Single qubit index (must be 1 qubit)
- **Returns:** `pyqpanda3.QCircuit`

**Example:**
```python
from pyvqnet.qnn.pq3.template import RotCircuit
import pyqpanda3.core as pq
from pyvqnet import tensor

param = tensor.QTensor([3, 4, 5])
c = RotCircuit(param, 0)
print(c)
```

---

### CRotCircuit

```python
pyvqnet.qnn.pq3.template.CRotCircuit(para, control_qubits, rot_qubits)
```

**Description:**
Controlled arbitrary single-qubit rotation.

**Matrix:**
```
[ 1  0  0                  0                ]
[ 0  1  0                  0                ]
[ 0  0  e^{-i(φ+ω)/2}cos(θ/2)  -e^{i(φ-ω)/2}sin(θ/2) ]
[ 0  0  e^{-i(φ-ω)/2}sin(θ/2)   e^{i(φ+ω)/2}cos(θ/2) ]
```

**Parameters:**
- `para` - `np.ndarray` - `[φ, θ, ω]` parameters
- `control_qubits` - `List[int]` - Control qubit (must be 1 qubit)
- `rot_qubits` - `List[int]` - Target rotation qubit (must be 1 qubit)
- **Returns:** `pyqpanda3.QCircuit`

**Example:**
```python
from pyvqnet.qnn.pq3.template import CRotCircuit
import pyqpanda3.core as pq
import numpy as np

m_qlist = [0]
control_qlist = [1]
param = np.array([3, 4, 5])
cir = CRotCircuit(param, control_qlist, m_qlist)
print(cir)
```

---

## Multi-Qubit Gates

### CSWAPcircuit

```python
pyvqnet.qnn.pq3.template.CSWAPcircuit(qubits)
```

**Description:**
Controlled-SWAP gate. The **first** qubit in the list is the control qubit.

**Matrix:**
```
8x8 matrix: diagonal with [1, 1, 1, 1, 1, 1, 1, 1]
except last entry is -1? Actually:
CSWAP = I_controls ⊕ SWAP when control=1
```

**Parameters:**
- `qubits` - `List[int]` - Qubit indices. First qubit is control, total length must be 3.
- **Returns:** `pyqpanda3.QCircuit`

**Example:**
```python
from pyvqnet.qnn.pq3 import CSWAPcircuit
import pyqpanda3.core as pq

m_qlist = range(3)
c = CSWAPcircuit([m_qlist[1], m_qlist[2], m_qlist[0]])  # control is last here
print(c)
```

---

### Controlled_Hadamard

```python
pyvqnet.qnn.pq3.template.Controlled_Hadamard(qubits)
```

**Description:**
Controlled-Hadamard (CH) gate.

**Matrix:**
```
[ 1  0    0      0     ]
[ 0  1    0      0     ]
[ 0  0  1/√2  1/√2    ]
[ 0  0  1/√2 -1/√2    ]
```

**Parameters:**
- `qubits` - `List[int]` - [control_qubit, target_qubit]
- **Returns:** `pyqpanda3.QCircuit`

**Example:**
```python
import pyqpanda3.core as pq
from pyvqnet.qnn.pq3 import Controlled_Hadamard

machine = pq.CPUQVM()
qubits = range(2)
cir = Controlled_Hadamard(qubits)
print(cir)
```

---

### CCZ

```python
pyvqnet.qnn.pq3.template.CCZ(qubits)
```

**Description:**
Controlled-Controlled-Z gate (two controls, one target).

**Matrix:**
8x8 diagonal matrix with all +1 except the last entry which is -1.

**Parameters:**
- `qubits` - `List[int]` - Three qubit indices
- **Returns:** `pyqpanda3.QCircuit`

**Example:**
```python
import pyqpanda3.core as pq
from pyvqnet.qnn.pq3 import CCZ

machine = pq.CPUQVM()
qubits = range(3)
cir = CCZ(qubits)
```

---

## Fermionic Operators

### FermionicSingleExcitation

```python
pyvqnet.qnn.pq3.template.FermionicSingleExcitation(
    weight,
    wires,
    qubits
)
```

**Description:**
Fermionic single excitation operator exponentiated for coupled cluster:
`Û_pr(θ) = exp{ θ (c_p† c_r - H.c.) }`

Uses Jordan-Wigner transformation.

**Parameters:**
- `weight` - `float` - Variational parameter θ
- `wires` - `List[int]` - Subset of qubit indices from r to p. Minimum length 2. First index = r, last = p. CNOT gates handle parity between them.
- `qubits` - `List[int]` - All qubits in the system
- **Returns:** `pyqpanda3.QCircuit`

**Example:**
```python
from pyvqnet.qnn.pq3 import FermionicSingleExcitation
import pyqpanda3.core as pq

machine = pq.CPUQVM()
qlists = range(3)
cir = FermionicSingleExcitation(0.5, [1, 0, 2], qlists)
```

---

### FermionicDoubleExcitation

```python
pyvqnet.qnn.pq3.template.FermionicDoubleExcitation(
    weight,
    wires1,
    wires2,
    qubits
)
```

**Description:**
Fermionic double excitation operator exponentiated for coupled cluster:
`Û_pqrs(θ) = exp{ θ (c_p† c_q† c_r c_s - H.c.) }`

Uses Jordan-Wigner transformation as described in arXiv:1805.04340.

**Parameters:**
- `weight` - `float` - Variational parameter θ
- `wires1` - `List[int]` - First interval [s, r] of occupied orbitals
- `wires2` - `List[int]` - Second interval [q, p] of empty orbitals
- `qubits` - `List[int]` - All qubits in the system
- **Returns:** `pyqpanda3.QCircuit`

**Example:**
```python
from pyvqnet.qnn.pq3 import FermionicDoubleExcitation
import pyqpanda3.core as pq

machine = pq.CPUQVM()
qlists = range(5)
weight = 1.5
cir = FermionicDoubleExcitation(
    weight,
    wires1=[0, 1],
    wires2=[2, 3, 4],
    qubits=qlists
)
```

---

### UCCSD

```python
pyvqnet.qnn.pq3.template.UCCSD(
    weights,
    wires,
    s_wires,
    d_wires,
    init_state,
    qubits
)
```

**Description:**
Unitary Coupled Cluster with Single and Double excitations (UCCSD) ansatz, commonly used for VQE quantum chemistry simulations. Uses first-order Trotter approximation.

**Formula:**
```
Û(θ) = Π_{p>r} exp{ θ_pr (c_p† c_r - H.c.) }
       Π_{p>q>r>s} exp{ θ_pqrs (c_p† c_q† c_r c_s - H.c.) }
```

**Parameters:**
- `weights` - `QTensor` - Shape `[len(s_wires) + len(d_wires)]` - Parameters for single and double excitations
- `wires` - `List[int]` - All qubit indices that play a role
- `s_wires` - `List[List[int]]` - List of single excitation wires, each `[r, ..., p]`
- `d_wires` - `List[List[List[int]]]` - List of double excitations, each is `[[s, ..., r], [q, ..., p]]`
- `init_state` - `List[int]` - Occupation number vector for Hartree-Fock reference
- `qubits` - `List[int]` - All qubit indices in the system
- **Returns:** `pyqpanda3.QCircuit`

**Example:**
```python
import pyqpanda3.core as pq
from pyvqnet.tensor import tensor
from pyvqnet.qnn.pq3 import UCCSD

machine = pq.CPUQVM()
qlists = range(6)
weight = tensor.zeros([8])
cir = UCCSD(
    weight,
    wires=[0, 1, 2, 3, 4, 5],
    s_wires=[[0, 1, 2], [0, 1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4, 5]],
    d_wires=[[[0, 1], [2, 3]], [[0, 1], [2, 3, 4, 5]], [[0, 1], [3, 4]], [[0, 1], [4, 5]]],
    init_state=[1, 1, 0, 0, 0, 0],
    qubits=qlists
)
```

---

## Quantum Pooling

### QuantumPoolingCircuit

```python
pyvqnet.qnn.pq3.template.QuantumPoolingCircuit(
    sources_wires,
    sinks_wires,
    params,
    qubits
)
```

**Description:**
Quantum pooling circuit for downsampling to reduce qubit count. Pairs qubits, applies a generalized two-qubit unitary to each pair, then ignores one qubit from each pair for the rest of the network.

**Parameters:**
- `sources_wires` - `List[int]` - Source qubits to be discarded after pooling
- `sinks_wires` - `List[int]` - Target qubits to keep after pooling
- `params` - `QTensor` - Parameters for the pooling unitary
- `qubits` - `List[int]` - All qubit indices
- **Returns:** `pyqpanda3.QCircuit`

**Example:**
```python
from pyvqnet.qnn.pq3.template import QuantumPoolingCircuit
import pyqpanda3.core as pq
from pyvqnet import tensor

qlists = range(4)
p = tensor.full([6], 0.35)
cir = QuantumPoolingCircuit([0, 1], [2, 3], p, qlists)
print(cir)
```

---

## Ansatz Templates

### HardwareEfficientAnsatz

```python
pyvqnet.qnn.pq3.ansatz.HardwareEfficientAnsatz(
    qubits,
    single_rot_gate_list,
    entangle_gate="CNOT",
    entangle_rules='linear',
    depth=1
)
```

**Description:**
Hardware Efficient Ansatz implementation as described in "Hardware-efficient Variational Quantum Eigensolver for Small Molecules" (arXiv:1704.05018). Call `create_ansatz()` to get the circuit.

**Parameters:**
- `qubits` - `List[int]` - Qubit indices
- `single_rot_gate_list` - `List[str]` - List of single-qubit rotation gates per qubit. Supports "rx", "ry", "rz" (case-insensitive)
- `entangle_gate` - `str` - Non-parameteric entanglement gate: "CNOT" or "CZ", default: "CNOT"
- `entangle_rules` - `str` - How entanglement gates are applied:
  - `"linear"` - Entangle adjacent qubits sequentially
  - `"all"` - Entangle every pair of qubits
  - default: `"linear"`
- `depth` - `int` - Ansatz depth (number of layers), default: 1

**Methods:**
- `create_ansatz(params)` - Returns the quantum circuit with given parameters

**Example:**
```python
import pyqpanda3.core as pq
from pyvqnet.tensor import QTensor, tensor
from pyvqnet.qnn.pq3.ansatz import HardwareEfficientAnsatz

machine = pq.CPUQVM()
qlist = range(4)
c = HardwareEfficientAnsatz(
    qlist,
    ["rx", "RY", "rz"],
    entangle_gate="cnot",
    entangle_rules="linear",
    depth=1
)
w = tensor.ones([c.get_para_num()])
cir = c.create_ansatz(w)
print(cir)
```

---

### BasicEntanglerTemplate

```python
pyvqnet.qnn.pq3.template.BasicEntanglerTemplate(
    weights=None,
    num_qubits=1,
    rotation=pyqpanda.RX
)
```

**Description:**
Basic entangling template: each layer consists of single-qubit rotations followed by CNOT gates in a chain (or ring) connecting adjacent qubits. The number of layers is determined by the first dimension of the weights. Call `create_circuit(qubits)` to get the circuit.

**Parameters:**
- `weights` - `np.ndarray` - Shape `(L, len(qubits))`. Each weight is a parameter for a rotation gate. Default: `None` uses `(1, 1)` random normal.
- `num_qubits` - `int` - Number of qubits, default: 1
- `rotation` - Single-parameter single-qubit gate class, default: `pyqpanda.RX`

**Methods:**
- `create_circuit(qubits)` - Builds and returns the quantum circuit
- `compute_circuit()` - Precomputes the circuit structure

**Example:**
```python
import pyqpanda3.core as pq
import numpy as np
from pyvqnet.qnn.pq3 import BasicEntanglerTemplate

np.random.seed(42)
num_qubits = 5
shape = [1, num_qubits]
weights = np.random.random(size=shape)

machine = pq.CPUQVM()
qubits = range(num_qubits)
circuit = BasicEntanglerTemplate(weights=weights, num_qubits=num_qubits, rotation=pq.RZ)
result = circuit.compute_circuit()
cir = circuit.create_circuit(qubits)
circuit.print_circuit(qubits)
```

---

### StronglyEntanglingTemplate

```python
pyvqnet.qnn.pq3.template.StronglyEntanglingTemplate(
    weights=None,
    num_qubits=1,
    ranges=None
)
```

**Description:**
Strongly entangling template from "circuit-centric classifier design" (arXiv:1804.00633). Each layer has single-qubit rotations (3 parameters per qubit) followed by CNOT gates between every qubit `i` and `(i + r) mod M`.

**Parameters:**
- `weights` - `np.ndarray` - Shape `(L, M, 3)` where L = layers, M = num_qubits. Default: `None` creates random `(1, 1, 3)`.
- `num_qubits` - `int` - Number of qubits, default: 1
- `ranges` - `List[int]` - Range parameter for each layer. Default: `None` uses `r = l mod M` for layer l.
- **Returns:** Template object, call `create_circuit(qubits)` to get the circuit.

**Example:**
```python
from pyvqnet.qnn.pq3 import StronglyEntanglingTemplate
import pyqpanda3.core as pq
from pyvqnet.tensor import *
import numpy as np

np.random.seed(42)
num_qubits = 3
shape = [2, num_qubits, 3]
weights = np.random.random(size=shape)

machine = pq.CPUQVM()
qubits = range(num_qubits)
circuit = StronglyEntanglingTemplate(weights, num_qubits=num_qubits)
result = circuit.compute_circuit()
cir = circuit.create_circuit(qubits)
circuit.print_circuit(qubits)
```

---

### ComplexEntangelingTemplate

```python
pyvqnet.qnn.pq3.ComplexEntangelingTemplate(
    weights,
    num_qubits,
    depth
)
```

**Description:**
Complex entangling template with U3 single-qubit gates and CNOT entangling layers. From the same paper as StronglyEntanglingTemplate: https://arxiv.org/abs/1804.00633.

**Parameters:**
- `weights` - Parameters of shape `[depth, num_qubits, 3]`
- `num_qubits` - `int` - Number of qubits
- `depth` - `int` - Number of layers
- **Returns:** Template object, call `create_circuit(qubits)` to get the circuit

**Example:**
```python
from pyvqnet.qnn.pq3 import ComplexEntangelingTemplate
import pyqpanda3.core as pq
from pyvqnet.tensor import *

depth = 3
num_qubits = 8
shape = [depth, num_qubits, 3]
weights = randn(shape)

machine = pq.CPUQVM()
qubits = range(num_qubits)
circuit = ComplexEntangelingTemplate(weights, num_qubits=num_qubits, depth=depth)
result = circuit.create_circuit(qubits)
circuit.print_circuit(qubits)
```

---

### Quantum_Embedding

```python
pyvqnet.qnn.pq3.Quantum_Embedding(
    qubits,
    machine,
    num_repetitions_input,
    depth_input,
    num_unitary_layers,
    num_repetitions
)
```

**Description:**
Variational quantum embedding for encoding classical data into quantum states, as described in "Quantum embeddings for machine learning" (arXiv:2001.03622). Uses RZ-RY-RZ construction to create a variational circuit that embeds classical data.

**Parameters:**
- `qubits` - Number of qubits allocated from pyqpanda
- `machine` - Quantum VM from pyqpanda
- `num_repetitions_input` - Number of repetitions for encoding input
- `depth_input` - Input feature dimension
- `num_unitary_layers` - Number of repeated variational gates per submodule
- `num_repetitions` - Number of submodule repetitions

**Methods:**
- `compute_circuit()` - Returns the circuit, can be used as input to `QuantumLayer`

**Example:**
```python
from pyvqnet.qnn.pq3 import QpandaQCircuitVQCLite, Quantum_Embedding
from pyvqnet.tensor import tensor
import pyqpanda3.core as pq

depth_input = 2
num_repetitions = 2
num_repetitions_input = 2
num_unitary_layers = 2

local_machine = pq.CPUQVM()
nq = depth_input * num_repetitions_input
qubits = range(nq)
cubits = range(nq)

data_in = tensor.ones([12, depth_input])
data_in.requires_grad = True

qe = Quantum_Embedding(nq, local_machine, num_repetitions_input,
                        depth_input, num_unitary_layers, num_repetitions)
qlayer = QpandaQCircuitVQCLite(qe.compute_circuit, qe.param_num)

y = qlayer.forward(data_in)
y.backward()
print(data_in.grad)
```
