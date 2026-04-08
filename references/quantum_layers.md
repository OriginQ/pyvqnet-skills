# Quantum Layers API Reference

## QuantumLayer

Also aliased as: `QuantumLayerV2`, `QpandaQCircuitVQCLayerLite`

```python
pyvqnet.qnn.pq3.quantumlayer.QuantumLayer(
    qprog_with_measure,
    para_num,
    diff_method: str = "parameter_shift",
    delta: float = 0.01,
    dtype=None,
    name=""
)
```

**Description:**
Variational quantum layer abstraction for PyQPanda3. Simulates a parameterized quantum circuit and returns measurement results. Inherits from VQNet's gradient computation module, supports gradient calculation via parameter-shift or finite-difference methods. Can be used for training variational quantum circuits or embedding quantum circuits into hybrid classical-quantum models.

**Parameters:**
- `qprog_with_measure` - User-defined quantum circuit function that returns measurement result. **MUST** have signature `qprog_with_measure(input, param)` where:
  - `input` - 1D classical input data, `None` if no input
  - `param` - 1D array of trainable variational parameters
  - **MUST** return `np.ndarray` or list of measurement expectation values
- `para_num` - `int` - Number of trainable parameters
- `diff_method` - `str` - Gradient calculation method: "parameter_shift" or "finite_diff", default: "parameter_shift"
- `delta` - `float` - Step size for finite difference gradient, default: 0.01
- `dtype` - Parameter data type, default: `None` (uses `kfloat32`)
- `name` - Module name, default: ""
- **Returns:** Quantum layer module

**Example:**
```python
from pyvqnet.qnn.pq3.measure import ProbsMeasure
from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
from pyvqnet.tensor import QTensor, ones
import pyqpanda3.core as pq

def pqctest(input, param):
    num_of_qubits = 4
    m_machine = pq.CPUQVM()
    qubits = range(num_of_qubits)
    circuit = pq.QCircuit()

    circuit << pq.H(0) << pq.H(1) << pq.H(2) << pq.H(3)
    circuit << pq.RZ(0, input[0]) << pq.RZ(1, input[1])
    circuit << pq.RZ(2, input[2]) << pq.RZ(3, input[3])
    circuit << pq.CNOT(0, 1) << pq.RZ(1, param[0]) << pq.CNOT(0, 1)
    circuit << pq.CNOT(1, 2) << pq.RZ(2, param[1]) << pq.CNOT(1, 2)
    circuit << pq.CNOT(2, 3) << pq.RZ(3, param[2]) << pq.CNOT(2, 3)

    prog = pq.QProg()
    prog << circuit
    rlt_prob = ProbsMeasure(m_machine, prog, [0, 2])
    return rlt_prob

pqc = QuantumLayer(pqctest, 3)

# Forward pass
input = QTensor([[1, 2, 3, 4], [4, 2, 2, 3], [3.0, 3, 2, 2]])
rlt = pqc(input)
print(rlt)

# Backward pass
grad = ones(rlt.data.shape) * 1000
rlt.backward(grad)
print(pqc.m_para.grad)
```

**Note:**
- Gradient computation with parameter-shift requires additional pyqpanda3 simulations. Computational complexity scales linearly with `para_num × batch_size × input_dim`.

---

## QpandaQProgVQCLayer

Also aliased as: `QuantumLayerV3`

```python
pyvqnet.qnn.pq3.quantumlayer.QpandaQProgVQCLayer(
    origin_qprog_func,
    para_num,
    qvm_type="cpu",
    pauli_str_dict=None,
    shots=1000,
    initializer=None,
    dtype=None,
    name=""
)
```

**Description:**
Submits a parameterized quantum circuit to local PyQPanda3 full-amplitude simulator for computation and trains circuit parameters. Supports batched data and uses parameter-shift rule for gradient estimation. For `CRX`, `CRY`, `CRZ` uses special gradient formulas from https://iopscience.iop.org/article/10.1088/1367-2630/ac2cb3, other gates use default parameter-shift.

**Parameters:**
- `origin_qprog_func` - Callable function returning `pyqpanda3.core.QProg`. **MUST** have signature `origin_qprog_func(input, param)`:
  - `input` - 1D classical input data
  - `param` - 1D array of parameters
  - **MUST** return `pyqpanda3.core.QProg`
- `para_num` - `int` - Number of parameters (1D)
- `qvm_type` - `str` - "cpu" or "gpu", default: "cpu"
- `pauli_str_dict` - `dict | list` - Dictionary/list of Pauli string expectations, default: `None`
- `shots` - `int` - Number of measurement shots, default: 1000
- `initializer` - Parameter initializer function, default: `None`
- `dtype` - Parameter data type, default: `None`
- `name` - Module name, default: ""
- **Returns:** `QuantumLayerV3` instance

**Example:**
```python
import pyqpanda3.core as pq
from pyvqnet.qnn.pq3.quantumlayer import QpandaQProgVQCLayer
from pyvqnet.utils.initializer import ones
from pyvqnet.tensor import QTensor

def qfun(input, param):
    m_qlist = range(3)
    cubits = range(3)
    measure_qubits = [0, 1, 2]
    m_prog = pq.QProg()
    cir = pq.QCircuit(3)

    cir << pq.RZ(m_qlist[0], input[0])
    cir << pq.RX(m_qlist[2], input[2])
    qcir = pq.RX(m_qlist[1], param[1]).control(m_qlist[0])
    cir << qcir
    qcir = pq.RY(m_qlist[0], param[2]).control(m_qlist[1])
    cir << qcir
    cir << pq.RY(m_qlist[0], input[1])
    qcir = pq.RZ(m_qlist[0], param[3]).control(m_qlist[1])
    cir << qcir
    m_prog << cir

    for idx, ele in enumerate(measure_qubits):
        m_prog << pq.measure(m_qlist[ele], cubits[idx])
    return m_prog

layer = QpandaQProgVQCLayer(qfun, 4, "cpu", initializer=ones)
x = QTensor([[2.56, 1.2, -3]], requires_grad=True)
y = layer(x)
y.backward()
print(layer.m_para.grad.to_numpy())
print(x.grad.to_numpy())
```

---

## QuantumBatchAsyncQcloudLayer

```python
pyvqnet.qnn.pq3.quantumlayer.QuantumBatchAsyncQcloudLayer(
    origin_qprog_func,
    qcloud_token,
    para_num,
    pauli_str_dict=None,
    shots=1000,
    initializer=None,
    dtype=None,
    name="",
    diff_method="parameter_shift",
    submit_kwargs={},
    query_kwargs={}
)
```

**Description:**
Runs variational quantum circuits on Origin Quantum QCloud real quantum hardware. Submits parameterized circuits asynchronously and obtains measurement results. Supports `random_coordinate_descent` gradient method where only a single random parameter is updated per step (reference: https://arxiv.org/abs/2311.00088).

**Parameters:**
- `origin_qprog_func` - Returns `pyqpanda3.core.QProg`, signature: `origin_qprog_func(input, param)`:
  - `input` - 1D or 2D classical input (first dimension is batch)
  - `param` - 1D array of parameters
  - If `pauli_str_dict` is None, the QProg must contain measurements
- `qcloud_token` - `str` - API token from https://qcloud.originqc.com.cn/
- `para_num` - `int` - Number of parameters
- `pauli_str_dict` - `dict | list` - Pauli expectations, default: `None`. If provided, computes expectation values.
- `shots` - `int` - Measurement shots, default: 1000
- `initializer` - Parameter initializer, default: `None` (0~2π normal distribution)
- `dtype` - Data type, default: `None` (kfloat32)
- `name` - Module name, default: ""
- `diff_method` - "parameter_shift" or "random_coordinate_descent", default: "parameter_shift"
- `submit_kwargs` - Additional kwargs for circuit submission:
  - Default: `{"chip_id": "origin_wukong", "is_amend": True, "is_mapping": True, "is_optimization": True, "compile_level": 3, "default_task_group_size": 200, "test_qcloud_fake": False}`
  - Set `test_qcloud_fake: True` to use local CPUQVM simulation for testing
- `query_kwargs` - Additional kwargs for result querying:
  - Default: `{"timeout": 1, "total_timeout": 60, "print_query_info": True, "sub_circuits_split_size": 1}`
  - `total_timeout` - Maximum wait time in seconds (default 60)
- **Returns:** Quantum layer that runs on QCloud

**Example:**
```python
import pyqpanda3.core as pq
from pyvqnet.qnn.pq3.quantumlayer import QuantumBatchAsyncQcloudLayer
from pyvqnet.tensor import QTensor
import os

token = os.getenv("ORIGINQC_API_KEY")

def qfun(input, param):
    measure_qubits = [0, 2]
    m_qlist = range(6)
    cir = pq.QCircuit(6)
    cir << pq.RZ(m_qlist[0], input[0]) << pq.CNOT(m_qlist[0], m_qlist[1])
    cir << pq.RY(m_qlist[1], param[0]) << pq.CNOT(m_qlist[0], m_qlist[2])
    cir << pq.RZ(m_qlist[1], input[1]) << pq.RY(m_qlist[2], param[1])
    cir << pq.H(m_qlist[2])
    m_prog = pq.QProg(cir)

    for idx, ele in enumerate(measure_qubits):
        m_prog << pq.measure(m_qlist[ele], m_qlist[idx])
    return m_prog

layer = QuantumBatchAsyncQcloudLayer(
    qfun, token, 2,
    submit_kwargs={"test_qcloud_fake": True}  # Fake mode for testing
)
x = QTensor([[0.56, 1.2], [0.56, 1.2]], requires_grad=True)
y = layer(x)
y.backward()
print(layer.m_para.grad)
print(x.grad)
```

**Note:**
- Computational cost scales with `para_num × batch_size × input_dim` due to extra gradient evaluations
- Default timeout is 60 seconds - increase `total_timeout` in `query_kwargs` if QCloud is busy

---

## QuantumLayerAdjoint

```python
pyvqnet.qnn.pq3.quantumlayer.QuantumLayerAdjoint(
    pq3_vqc_circuit,
    param_num,
    pauli_dicts,
    dtype=None,
    name=""
)
```

**Description:**
Uses the adjoint method to compute gradients of parameters with respect to Hamiltonian expectation, using PyQPanda3's `VQCircuit` interface. Supports batched input and multiple Hamiltonian outputs.

**Parameters:**
- `pq3_vqc_circuit` - Custom function that returns a `pyqpanda3.vqcircuit.VQCircuit`. **MUST** have signature `pq3_vqc_circuit(x, param)` where:
  - `x` - Input (1D array/list)
  - `param` - Parameters (1D array/list)
  - User must use `vqc.set_Param()` inside the function
- `param_num` - `int` - Number of parameters
- `pauli_dicts` - Expected observables, can be a list of dictionaries
- `dtype` - Parameter data type (kfloat32 or kfloat64), default: `None` (kfloat32)
- `name` - Interface name
- **Returns:** `QuantumLayerAdjoint` instance

**Notes:**
- You MUST use gates from the `VQCircuit` interface to build your circuit
- Only a limited set of gates are currently supported. Unsupported gates will throw an exception.

**Example:**
```python
from pyvqnet.qnn.pq3 import QuantumLayerAdjoint
from pyvqnet.tensor import randn
from pyqpanda3.vqcircuit import VQCircuit
import pyqpanda3 as pq3

l = 3
n = 7
def pqctest(x, param):
    vqc = VQCircuit()
    vqc.set_Param([len(param) + len(x)])
    w_offset = len(x)
    for j in range(len(x)):
        vqc << pq3.core.RX(j, vqc.Param([j]))
    for j in range(l):
        for i in range(n - 1):
            vqc << pq3.core.CNOT(i, i + 1)
        for i in range(n):
            vqc << pq3.core.RX(i, vqc.Param([w_offset + 3 * n * j + i]))
            vqc << pq3.core.RZ(i, vqc.Param([w_offset + 3 * n * j + i + n]))
            vqc << pq3.core.RY(i, vqc.Param([w_offset + 3 * n * j + i + 2 * n]))
    return vqc

Xn_string = ' '.join([f'X{i}' for i in range(n)])
pauli_dict = {Xn_string: 1.}

layer = QuantumLayerAdjoint(pqctest, 3 * l * n, pauli_dict)
x = randn([2, 5])
x.requires_grad = True
y = layer(x)
y.backward()
print(layer.m_para.grad)
print(x.grad)
```

---

## grad

```python
pyvqnet.qnn.pq3.quantumlayer.grad(
    quantum_prog_func,
    input_params,
    *args
)
```

**Description:**
Computes gradients for user-defined parameterized quantum circuits using the parameter-shift method.

**Parameters:**
- `quantum_prog_func` - User's quantum circuit function that computes expectation
- `input_params` - Coordinates of parameters for which to compute gradient
- `*args` - Additional arguments passed to `quantum_prog_func`
- **Returns:** Gradient array with shape `[num_of_parameters, num_of_output]`

**Example:**
```python
from pyvqnet.qnn.pq3 import grad, ProbsMeasure
import pyqpanda3.core as pq

def pqctest(param):
    machine = pq.CPUQVM()
    qubits = range(2)
    circuit = pq.QCircuit(2)

    circuit << pq.RX(qubits[0], param[0])
    circuit << pq.RY(qubits[1], param[1])
    circuit << pq.CNOT(qubits[0], qubits[1])
    circuit << pq.RX(qubits[1], param[2])

    prog = pq.QProg()
    prog << circuit
    EXP = ProbsMeasure(machine, prog, [1])
    return EXP

g = grad(pqctest, [0.1, 0.2, 0.3])
print(g)
exp = pqctest([0.1, 0.2, 0.3])
print(exp)
```
