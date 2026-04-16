# Quantum Layers API Reference

> 来源: VQNET2.0-tutorial/source/rst/qnn_pq3.rst + qnn.rst
> **重要**: 所有示例代码均来自官方文档，可直接运行。

---

## QuantumLayer (pyqpanda3)

```python
pyvqnet.qnn.pq3.quantumlayer.QuantumLayer(
    qprog_with_measure,
    para_num,
    diff_method="parameter_shift",
    delta=0.01,
    dtype=None,
    name=""
)
```

**别名**: `QuantumLayerV2`, `QpandaQCircuitVQCLayerLite`

**参数**:
- `qprog_with_measure` - 量子电路函数，**签名必须为 `(input, param)`**
- `para_num` - 可训练参数个数
- `diff_method` - 梯度方法: "parameter_shift" 或 "finite_diff"
- `delta` - 有限差分步长

**关键签名要求**:
```python
def qprog_with_measure(input, param):
    # input: 一维经典输入数据
    # param: 一维变分参数
    # 返回: np.ndarray 或 list（测量结果）
```

**示例**:
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

input = QTensor([[1, 2, 3, 4], [4, 2, 2, 3], [3.0, 3, 2, 2]])
rlt = pqc(input)
print(rlt)

grad = ones(rlt.data.shape) * 1000
rlt.backward(grad)
print(pqc.m_para.grad)
```

---

## QpandaQProgVQCLayer

**别名**: `QuantumLayerV3`

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

**参数**:
- `origin_qprog_func` - 返回 `pyqpanda3.core.QProg` 的函数
- `para_num` - 参数数量
- `qvm_type` - "cpu" 或 "gpu"
- `pauli_str_dict` - Pauli 期望字典，如 `{'Z0 X1': 10}`
- `shots` - 测量次数

**示例**:
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

**真机运行** - 提交量子电路到本源量子云。

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

**参数**:
- `qcloud_token` - 从 https://qcloud.originqc.com.cn/ 获取的 API Token
- `submit_kwargs` - 提交参数，默认: `{"chip_id": "origin_wukong", ...}`
- `query_kwargs` - 查询参数，默认: `{"timeout": 1, "total_timeout": 60}`
- 设置 `test_qcloud_fake: True` 使用本地模拟测试

**示例**:
```python
import pyqpanda3.core as pq
from pyvqnet.qnn.pq3.quantumlayer import QuantumBatchAsyncQcloudLayer
from pyvqnet.tensor import QTensor
import os

token = os.getenv("QCLOUD_TOKEN")  # 不要硬编码！

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
    submit_kwargs={"test_qcloud_fake": True}  # 测试模式
)
x = QTensor([[0.56, 1.2], [0.56, 1.2]], requires_grad=True)
y = layer(x)
y.backward()
print(layer.m_para.grad)
print(x.grad)
```

---

## QuantumLayerAdjoint

使用 adjoint 方法计算梯度。

```python
pyvqnet.qnn.pq3.quantumlayer.QuantumLayerAdjoint(
    pq3_vqc_circuit,
    param_num,
    pauli_dicts,
    dtype=None,
    name=""
)
```

**示例**:
```python
from pyvqnet.qnn.pq3 import QuantumLayerAdjoint
from pyvqnet import tensor
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
x = tensor.randn([2, 5])
x.requires_grad = True
y = layer(x)
y.backward()
print(layer.m_para.grad)
print(x.grad)
```

---

## QLinear (量子全连接)

```python
pyvqnet.qnn.qlinear.QLinear(input_channels, output_channels, machine="CPU")
```

**示例**:
```python
from pyvqnet.tensor import QTensor
from pyvqnet.qnn.qlinear import QLinear

params = [[0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
          [1.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
          [1.37454012, 1.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
          [1.37454012, 1.95071431, 1.73199394, 1.59865848, 0.15601864, 0.15599452]]

m = QLinear(6, 2)
input = QTensor(params, requires_grad=True)
output = m(input)
output.backward()
print(output)
```

---

## QConv (量子卷积)

```python
pyvqnet.qnn.qcnn.qconv.QConv(
    input_channels,
    output_channels,
    quantum_number,
    stride=(1, 1),
    padding=(0, 0),
    kernel_initializer=normal,
    machine="CPU",
    dtype=None,
    name=""
)
```

**示例**:
```python
from pyvqnet.tensor import tensor
from pyvqnet.qnn.qcnn.qconv import QConv

x = tensor.ones([1, 3, 4, 4])
layer = QConv(input_channels=3, output_channels=2, quantum_number=4, stride=(2, 2))
y = layer(x)
print(y)
```

---

## 测量函数

### ProbsMeasure
概率测量。

```python
from pyvqnet.qnn.pq3.measure import ProbsMeasure

ProbsMeasure(machine, prog, measure_qubits)
```

### expval
Pauli 期望值测量。

```python
from pyvqnet.qnn.pq3 import expval

expval(machine, prog, pauli_str_dict)
```

---

## 量子门模板

### AmplitudeEmbeddingCircuit
振幅编码。

```python
from pyvqnet.qnn.pq3.template import AmplitudeEmbeddingCircuit
import numpy as np
import pyqpanda3.core as pq

input_feat = np.array([2.2, 1, 4.5, 3.7])
qlist = range(3)
cir = AmplitudeEmbeddingCircuit(input_feat, qlist)
```

### AngleEmbeddingCircuit
角度编码。

```python
from pyvqnet.qnn.pq3.template import AngleEmbeddingCircuit
import numpy as np

m_qlist = range(2)
input_feat = np.array([2.2, 1])
C = AngleEmbeddingCircuit(input_feat, m_qlist, 'X')
```

### RotCircuit
任意单量子比特旋转。

```python
from pyvqnet.qnn.pq3.template import RotCircuit
from pyvqnet import tensor

param = tensor.QTensor([3, 4, 5])
c = RotCircuit(param, 1)
```

### HardwareEfficientAnsatz
硬件高效 ansatz。

```python
from pyvqnet.qnn.pq3.ansatz import HardwareEfficientAnsatz

hea = HardwareEfficientAnsatz(
    qubits=range(4),
    single_rot_gate_list=["RX", "RY", "RZ"],
    entangle_gate="CNOT",
    entangle_rules="linear",
    depth=2
)
cir = hea.create_ansitz()
```

---

## 常见问题

1. **参数顺序错误**: 函数签名必须是 `(input, param)`，不是 `(param, input)`
2. **忘记返回测量结果**: 函数必须返回 `np.ndarray` 或 `list`
3. **QCloud Token**: 使用 `os.getenv("QCLOUD_TOKEN")`，不要硬编码
4. **梯度计算开销**: parameter-shift 需要额外运行 `para_num × batch_size × input_dim` 次电路

---

**Version**: VQNet 2.0
**Source**: VQNET2.0-tutorial/source/rst/qnn_pq3.rst