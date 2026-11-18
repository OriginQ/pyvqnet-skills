# VQC 自动微分 API Reference

> **重要**: 所有示例代码均来自官方文档。VQC 模块不依赖 pyqpanda3 等外部量子线路库，使用 VQNet 内置的自动微分模拟。

---

## QMachine 模拟器

```python
pyvqnet.qnn.vqc.QMachine(
    num_wires,
    dtype=pyvqnet.kcomplex64,
    grad_mode="",
    save_ir=False
)
```

**参数**:
- `num_wires` - 量子比特数
- `dtype` - 数据类型，默认 `kcomplex64`（对应参数 `kfloat32`）
- `grad_mode` - 梯度模式，可设为 "adjoint"

**关键**: 必须在每次 forward 开头调用 `reset_states(batchsize)`

**示例**:
```python
from pyvqnet.qnn.vqc import QMachine

qm = QMachine(4)
print(qm.states)
# [[[[[1.+0.j 0.+0.j]
#     [0.+0.j 0.+0.j]]
#    [[0.+0.j 0.+0.j]
#     [0.+0.j 0.+0.j]]]
#   [[[0.+0.j 0.+0.j]
#     [0.+0.j 0.+0.j]]
#    [[0.+0.j 0.+0.j]
#     [0.+0.j 0.+0.j]]]]]
```

---

## 量子门接口（函数形式）

### 单量子比特门

```python
from pyvqnet.qnn.vqc import hadamard, paulix, pauliy, pauliz, rx, ry, rz, QMachine
from pyvqnet.tensor import QTensor

qm = QMachine(4)

# 无参数门
hadamard(q_machine=qm, wires=1)
paulix(q_machine=qm, wires=1)
pauliy(q_machine=qm, wires=1)
pauliz(q_machine=qm, wires=1)

# 有参数门
rx(q_machine=qm, wires=1, params=QTensor([0.5]))
ry(q_machine=qm, wires=1, params=QTensor([0.5]))
rz(q_machine=qm, wires=1, params=QTensor([0.5]))
```

### 双量子比特门

```python
from pyvqnet.qnn.vqc import cnot, cz, swap, QMachine

qm = QMachine(4)
cnot(q_machine=qm, wires=[0, 1])
cz(q_machine=qm, wires=[0, 1])
swap(q_machine=qm, wires=[0, 1])
```

---

## 量子门类（Module 形式）

### 统一签名（2.18.2）

> **2.18.2 变更**: 所有量子门类统一为以下签名，`trainable` 作为第一个位置参数。

```python
# 所有门类统一签名（仅类名不同）：
X(trainable=False, init_params=None, wires=None, dtype=pyvqnet.kcomplex64, use_dagger=False)
# MultiControlledX 额外接受 control_values=None
# 门类：I, Hadamard, T, S, PauliX, PauliY, PauliZ, X1, RX, RY, RZ, CRX, CRY, CRZ,
# U1, U2, U3, CY, CNOT, CR, SWAP, CSWAP, CZ, RXX, RYY, RZZ, RZX, Toffoli,
# IsingXX, IsingYY, IsingZZ, IsingXY, PhaseShift, MultiRZ, SDG, TDG,
# ControlledPhaseShift, MultiControlledX
```

`trainable=True` 表示该门自带可训练参数；使用外部输入数据编码时设置为 `False` 并通过 `params=` 传入。可训练门的参数对象为 `gate.params`，反向传播后通过 `gate.params.grad` 读取梯度；也可用 `model.parameters()` 遍历所有可训练参数。

### Hadamard

```python
from pyvqnet.qnn.vqc import Hadamard, QMachine

device = QMachine(4)
layer = Hadamard(wires=0)
device.reset_states(1)
layer(q_machine=device)
print(device.states)
```

### RX / RY / RZ

```python
from pyvqnet.qnn.vqc import RX, RY, RZ, QMachine
from pyvqnet.tensor import QTensor

# 参数化旋转门 - 设置 trainable=True
device = QMachine(4)

# 可训练的 RX 门
layer = RX(trainable=True, wires=0)
device.reset_states(2)
layer(q_machine=device)

# 使用外部参数编码数据
encode = RZ(wires=0)
encode(params=QTensor([0.5]), q_machine=device)
```

### PauliX / PauliY / PauliZ

```python
from pyvqnet.qnn.vqc import PauliX, PauliY, PauliZ, QMachine

device = QMachine(4)
layer = PauliX(wires=0)
device.reset_states(1)
layer(q_machine=device)
```

### CNOT / CZ / SWAP

```python
from pyvqnet.qnn.vqc import CNOT, CZ, SWAP, QMachine

device = QMachine(4)
layer = CNOT(wires=[0, 1])
device.reset_states(1)
layer(q_machine=device)
```

---

## 测量

### Probability

```python
Probability(wires, name="")
```

```python
from pyvqnet.qnn.vqc import Probability, QMachine

# 概率测量
measure = Probability(wires=[0, 2])
device = QMachine(4)
device.reset_states(1)
prob = measure(q_machine=device)
print(prob)
```

**输出形状**: `Probability(wires=[0, 2])` 返回 `[batchsize, 2**len(wires)]`；`MeasureAll(obs={"Z0": 1})` 返回 `[batchsize, 1]`；若需多个独立期望值，传入 `obs=[{"Z0": 1}, {"Z1": 1}]`，返回 `[batchsize, 2]`（多键字典会被视为一个组合观测量）。

---

## Ansatz 模板

### VQC_HardwareEfficientAnsatz

```python
VQC_HardwareEfficientAnsatz(
    n_qubits,
    single_rot_gate_list,
    entangle_gate="CNOT",
    entangle_rules="linear",
    depth=1,
    initial=None,
    dtype=None
)
```

```python
from pyvqnet.qnn.vqc import VQC_HardwareEfficientAnsatz

ansatz = VQC_HardwareEfficientAnsatz(
    4,  # num_wires
    ["rx", "RY", "rz"],  # 旋转门列表
    entangle_gate="cnot",
    entangle_rules="linear",
    depth=2
)
ansatz(q_machine=device)
```

### VQC_BasicEntanglerTemplate

```python
VQC_BasicEntanglerTemplate(num_layers=1, num_qubits=1, rotation="RX", initial=None, dtype=None)
```

由每个量子位上的单参数单量子位旋转层，后跟一个闭合链或环组合的多个 CNOT 门构成。CNOT 门环将每个量子位与其邻居连接起来，最后一个量子位被认为是第一个量子位的邻居。

**参数**:
- `num_layers` - 量子比特线路层数
- `num_qubits` - 量子比特数，默认为 1
- `rotation` - 单参数单量子比特门，默认 `"RX"`
- `initial` - 初始化所有参数逻辑门的参数，默认 None（随机初始化）
- `dtype` - 参数数据类型，默认 None（使用 float32）

**示例**:
```python
from pyvqnet.nn import Module
from pyvqnet.qnn.vqc import VQC_BasicEntanglerTemplate
from pyvqnet.qnn.vqc import Probability, QMachine
from pyvqnet import tensor


class QM(Module):
    def __init__(self, name=""):
        super().__init__(name)
        self.ansatz = VQC_BasicEntanglerTemplate(
            2,
            4,
            "rz",
            initial=tensor.ones([1, 1])
        )
        self.measure = Probability(wires=[0, 2])
        self.device = QMachine(4)

    def forward(self, x, *args, **kwargs):
        self.ansatz(q_machine=self.device)
        return self.measure(q_machine=self.device)

bz = 1
inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
qlayer = QM()
y = qlayer(inputx)
y.backward()
print(y)
# [[1.0000002 0.        0.        0.       ]]
```

### VQC_StronglyEntanglingTemplate

```python
VQC_StronglyEntanglingTemplate(num_layers=1, num_qubits=1, ranges=None, initial=None, dtype=None)
```

由单个量子比特旋转和纠缠器组成的层，参考 `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`_。

**参数**:
- `num_layers` - 重复层数，默认 1
- `num_qubits` - 量子比特数，默认 1
- `ranges` - 决定每个后续层的范围超参数的序列。默认 None，对第 l 层和 M 个量子比特使用 r = l mod M。
- `initial` - 初始化所有参数逻辑门的参数，默认 None（随机初始化）
- `dtype` - 参数数据类型，默认 None（使用 float32）

**示例**:
```python
from pyvqnet.nn import Module
from pyvqnet.qnn.vqc import VQC_StronglyEntanglingTemplate
from pyvqnet.qnn.vqc import Probability, QMachine
from pyvqnet import tensor


class QM(Module):
    def __init__(self, name=""):
        super().__init__(name)
        self.ansatz = VQC_StronglyEntanglingTemplate(
            2,
            4,
            None,
            initial=tensor.ones([1, 1])
        )
        self.measure = Probability(wires=[0, 1])
        self.device = QMachine(4)

    def forward(self, x, *args, **kwargs):
        self.ansatz(q_machine=self.device)
        return self.measure(q_machine=self.device)

bz = 1
inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
qlayer = QM()
y = qlayer(inputx)
y.backward()
print(y)
# [[0.3745951 0.154298  0.059156  0.4119509]]
```

### VQC_AmplitudeEmbedding

```python
VQC_AmplitudeEmbedding(input_feature, q_machine)
```

将 2^n 个特征编码为 `q_machine` n 量子比特的振幅向量。

**参数**:
- `input_feature` - 表示参数的 numpy 数组或 QTensor
- `q_machine` - 量子虚拟机设备

**示例**:
```python
from pyvqnet.qnn.vqc import VQC_AmplitudeEmbedding, QMachine
from pyvqnet.tensor import QTensor

qm = QMachine(3)
VQC_AmplitudeEmbedding(QTensor([3.2, -2, -2, 0.3, 12, 0.1, 2, -1]), q_machine=qm)
print(qm.states)
```

### VQC_CRotCircuit

```python
VQC_CRotCircuit(para, control_wire, rot_wire, q_machine)
```

在 `q_machine` 的 `rot_wire` 以及受控比特 `control_wire` 上使用 `para` 进行受控 Rot 操作符。

**参数**:
- `para` - 表示参数 `[φ, θ, ω]` 的数组
- `control_wire` - 控制量子比特索引
- `rot_wire` - Rot 量子比特索引
- `q_machine` - 量子虚拟机设备

**示例**:
```python
from pyvqnet.tensor import QTensor
from pyvqnet.qnn.vqc import VQC_CRotCircuit
from pyvqnet.qnn.vqc import QMachine, MeasureAll

p = QTensor([2, 3, 4.0])
qm = QMachine(2)
VQC_CRotCircuit(p, 0, 1, qm)
m = MeasureAll(obs={"Z0": 1})
exp = m(q_machine=qm)
print(exp)
# [[0.9999999]]
```

---

## 完整示例：混合 QNN 模型

```python
from pyvqnet.nn import Module, Linear, ModuleList
from pyvqnet.qnn.vqc import VQC_HardwareEfficientAnsatz, RZ
from pyvqnet.qnn.vqc import Probability, QMachine
from pyvqnet import tensor

class QM(Module):
    def __init__(self, name=""):
        super().__init__(name)
        self.linearx = Linear(4, 2)
        self.ansatz = VQC_HardwareEfficientAnsatz(
            4, ["rx", "RY", "rz"],
            entangle_gate="cnot",
            entangle_rules="linear",
            depth=2
        )
        self.encode1 = RZ(wires=0)
        self.encode2 = RZ(wires=1)
        self.measure = Probability(wires=[0, 2])
        self.device = QMachine(4)

    def forward(self, x, *args, **kwargs):
        # 必须重置状态！
        self.device.reset_states(x.shape[0])
        y = self.linearx(x)
        # 编码输入 - shape 必须是 [batchsize, 1]
        self.encode1(params=y[:, [0]], q_machine=self.device)
        self.encode2(params=y[:, [1]], q_machine=self.device)
        self.ansatz(q_machine=self.device)
        return self.measure(q_machine=self.device)

bz = 3
inputx = tensor.arange(1.0, bz*4+1).reshape([bz, 4])
inputx.requires_grad = True
qlayer = QM()
y = qlayer(inputx)
y.backward()
print(y)
```

---

## 带训练参数的版本

```python
from pyvqnet.nn import Module, Linear
from pyvqnet.qnn.vqc import VQC_HardwareEfficientAnsatz, RZ
from pyvqnet.qnn.vqc import Probability, QMachine
from pyvqnet import tensor

class QM(Module):
    def __init__(self, name=""):
        super().__init__(name)
        self.linearx = Linear(4, 2)
        self.ansatz = VQC_HardwareEfficientAnsatz(
            4, ["rx", "RY", "rz"],
            entangle_gate="cnot",
            depth=2
        )
        self.encode1 = RZ(wires=0)
        self.encode2 = RZ(wires=1)
        # 设置 trainable=True 来添加可训练参数
        self.vqc = RZ(trainable=True, wires=1)
        self.measure = Probability(wires=[0, 2])
        self.device = QMachine(4)

    def forward(self, x):
        self.device.reset_states(x.shape[0])
        y = self.linearx(x)
        self.encode1(params=y[:, [0]], q_machine=self.device)
        self.encode2(params=y[:, [1]], q_machine=self.device)
        self.vqc(q_machine=self.device)  # 使用可训练参数
        self.ansatz(q_machine=self.device)
        return self.measure(q_machine=self.device)
```

> VQC 模型与普通 `Module` 使用相同的训练流程：`loss = loss_fn(target, pred)` → `optimizer.zero_grad()` → `loss.backward()` → `optimizer._step()`。注意 VQNet 损失函数参数顺序是 `(标签, 预测值)`。

---

## 与 QuantumLayer (pyqpanda3) 的区别

| 特性 | VQC 模块 | QuantumLayer |
|------|----------|--------------|
| 后端 | VQNet 内置模拟 | pyqpanda3 |
| 梯度 | 自动微分 | parameter-shift |
| 批处理 | 需要 reset_states | 自动 |
| 量子门 | `pyvqnet.qnn.vqc.*` | `pyqpanda3.core.*` |

---

## 关键注意事项

1. **reset_states**: 每次 forward 必须调用 `device.reset_states(batchsize)`
2. **参数形状**: 编码参数必须是 `[batchsize, 1]`
3. **trainable**: 设置 `trainable=True` 添加可训练参数
4. **dtype**: 参数精度为 `kfloat32`，门内部为 `kcomplex64`

---

## 支持的量子门

函数形式为小写（`i()`、`hadamard()`、`rx()`、`cnot()` 等），类形式为首字母大写（`I`、`Hadamard`、`RX`、`CNOT` 等）。完整门类列表见上文 `### 统一签名（2.18.2）`。

---

## QuantumLayerAdjoint (adjoint gradient)

```python
QuantumLayerAdjoint(vqc_module: pyvqnet.nn.Module, name="")
```

```python
from pyvqnet.qnn.vqc import QuantumLayerAdjoint

adjoint_model = QuantumLayerAdjoint(vqc_module, name="")
```

记忆高效的梯度计算方法，基于伴随矩阵方法（adjoint method）。参考论文: `Efficient calculation of gradients in classical simulations of variational quantum algorithms <https://arxiv.org/abs/2009.02823>`_。

**参数**:
- `vqc_module` - 一个仅使用 `pyvqnet.qnn.vqc` 下量子线路接口搭建的 `pyvqnet.nn.Module` 实例
- `name` - 模块名称，默认 `""`

**注意**:
1. 仅支持单参数量子门的 VQC 模块
2. QMachine 必须使用伴随梯度模式：构造参数为 `grad_mode="adjoint"`
3. 支持输入数据梯度和可训练参数梯度
4. `QuantumAdjointLayer` 是 `QuantumLayerAdjoint` 的兼容别名

**示例**:
```python
from pyvqnet import tensor
from pyvqnet.qnn.vqc import (
    QuantumLayerAdjoint, QMachine,
    RX, RY, RZ, CNOT, T,
    VQC_HardwareEfficientAnsatz, MeasureAll
)
import pyvqnet

class QModel(pyvqnet.nn.Module):
    def __init__(self, num_wires, dtype, grad_mode=""):
        super(QModel, self).__init__()
        self._num_wires = num_wires
        self.qm = QMachine(num_wires, dtype=dtype, grad_mode=grad_mode)
        self.rx_layer = RX(trainable=False, wires=0)
        self.ry_layer = RY(trainable=False, wires=1)
        self.rz_layer = RZ(trainable=False, wires=1)
        self.rz_layer2 = RZ(trainable=True, wires=1)
        self.rot = VQC_HardwareEfficientAnsatz(
            6, ["rx", "RY", "rz"],
            entangle_gate="cnot", entangle_rules="linear", depth=5
        )
        self.cnot = CNOT(wires=[0, 1])
        self.tlayer = T(wires=1)
        self.measure = MeasureAll(obs={"X1": 1})

    def forward(self, x, *args, **kwargs):
        self.qm.reset_states(x.shape[0])
        self.rx_layer(params=x[:, [0]], q_machine=self.qm)
        self.cnot(q_machine=self.qm)
        self.ry_layer(params=x[:, [1]], q_machine=self.qm)
        self.tlayer(q_machine=self.qm)
        self.rz_layer(params=x[:, [2]], q_machine=self.qm)
        self.rz_layer2(q_machine=self.qm)
        self.rot(q_machine=self.qm)
        return self.measure(q_machine=self.qm)

input_x = tensor.QTensor([[0.1, 0.2, 0.3]])
input_x = tensor.broadcast_to(input_x, [40, 3])
input_x.requires_grad = True

qunatum_model = QModel(
    num_wires=6,
    dtype=pyvqnet.kcomplex64,
    grad_mode="adjoint"
)
adjoint_model = QuantumLayerAdjoint(qunatum_model, name="")
batch_y = adjoint_model(input_x)
batch_y.backward()
```

---

## CircuitGraph

```python
from pyvqnet.qnn.vqc import CircuitGraph

cg = CircuitGraph(allops, wires, trainable_params=None)
```

将量子电路表示为有向无环图（DAG），用于分析电路的因果结构、深度和参数化层级。底层依赖 `rustworkx` 库。

**参数**:
- `allops` - 量子操作列表（包含门和测量），按时间顺序排列
- `wires` - 量子比特编号的迭代器
- `trainable_params` - 可训练参数的索引集合（可选）

**关键属性**:
- `operations` - 电路中的操作列表（不包含可观测量）
- `observables` - 可观测量列表
- `graph` - `rustworkx.PyDiGraph` DAG 图
- `par_info` - 参数信息列表
- `num_wires` - 量子比特数
- `hash` - 电路的哈希值（基于序列化字符串）

**关键方法**:
- `get_depth()` - 计算电路深度（DAG最长路径）
- `parametrized_layers` - 参数化层级结构
- `iterate_parametrized_layers()` - 逐层遍历参数化层
- `ancestors(ops)` - 获取指定操作的祖先节点
- `descendants(ops)` - 获取指定操作的后代节点
- `has_path(a, b)` - 检查两节点间是否存在路径
- `hash` - 电路哈希；部分 2.18.2 构建中可能因缺少 `serialize` 实现而报错，示例只使用 `get_depth()` 与 `parametrized_layers`

**示例**:
```python
from pyvqnet.qnn.vqc import CircuitGraph, QMachine, RX, RY, CNOT, MeasureAll
from pyvqnet.tensor import QTensor

qm = QMachine(3)
qm.reset_states(1)
rx = RX(trainable=False, wires=0)
ry = RY(trainable=True, wires=1)
cnot = CNOT(wires=[0, 1])
measure = MeasureAll(obs={"Z0": 1, "Z1": 1})

rx(params=QTensor([0.5]), q_machine=qm)
cnot(q_machine=qm)
ry(q_machine=qm)
result = measure(q_machine=qm)

cg = CircuitGraph(
    qm.op_history,
    range(qm.num_wires),
    qm.train_params_indices
)
print(f"Circuit depth: {cg.get_depth()}")
for layer in cg.parametrized_layers:
    print(f"Layer ops: {layer.ops}, params: {layer.param_inds}")
```

---

## Block Encoding 方法

Block Encoding 是一类将经典矩阵编码到量子电路中的方法，使矩阵能以酉算子子块的形式作用于量子态。

### VQC_FABLE

```python
from pyvqnet.qnn.vqc import VQC_FABLE

fable = VQC_FABLE(wires)

# 前向传播
result = fable(q_machine, input_matrix, tol=0)
```

快速近似块编码（Fast Approximate Block Encoding），基于 `arXiv:2205.00081`。适用于具有特定结构的矩阵，可以在不降低精度的情况下简化电路。

**参数**:
- `wires` - 量子比特列表
- `input_matrix` - 输入矩阵（Tensor），元素值需在 [-1, 1] 范围内
- `tol` - 近似容差（默认 0）

**注意**: wires 数量必须为 `2n + 1`，其中 n 为矩阵维度对应的量子比特数。

**示例**:
```python
from pyvqnet.qnn.vqc import VQC_FABLE, QMachine
from pyvqnet.dtype import float_dtype_to_complex_dtype
from pyvqnet import QTensor
import numpy as np

A = QTensor(np.array([[0.1, 0.2], [0.3, 0.4]]))
qf = VQC_FABLE(list(range(3)))
qm = QMachine(3, dtype=float_dtype_to_complex_dtype(A.dtype))
qm.reset_states(1)
z1 = qf(qm, A, 0.001)
```

### VQC_LCU

```python
VQC_LCU(wires=None, check_hermitian=True)
```

```python
from pyvqnet.qnn.vqc import VQC_LCU

lcu = VQC_LCU(wires=None, check_hermitian=True)

# 前向传播
result = lcu(q_machine, input_matrix)
```

线性组合酉算子（Linear Combination of Unitaries），基于 `arXiv:1610.06546`。将 Hermitian 矩阵分解为 Pauli 算子的线性组合并编码到电路中。

**参数**:
- `wires` - 量子比特列表（可选；默认根据矩阵维度自动确定）
- `check_hermitian` - 是否检查输入矩阵为 Hermitian（默认 True）
- `input_matrix` - Hermitian 输入矩阵，维度需为 `2^n × 2^n`

**注意**: 输入矩阵必须是 Hermitian 的，且形状为 `[2^n, 2^n]`。

**示例**:
```python
from pyvqnet.qnn.vqc import VQC_LCU, QMachine
from pyvqnet.dtype import float_dtype_to_complex_dtype, kfloat64
from pyvqnet import QTensor

A = QTensor([
    [0.25, 0, 0, 0.75],
    [0, -0.25, 0.75, 0],
    [0, 0.75, 0.25, 0],
    [0.75, 0, 0, -0.25]
], dtype=kfloat64)
qf = VQC_LCU(list(range(3)))
qm = QMachine(3, dtype=float_dtype_to_complex_dtype(A.dtype))
qm.reset_states(2)
z1 = qf(qm, A)
```

### VQC_QSVT

```python
from pyvqnet.qnn.vqc import VQC_QSVT

qsvt = VQC_QSVT(A, angles, wires)

# 前向传播
result = qsvt(q_machine)
```

量子奇异值变换（Quantum Singular Value Transformation），基于 `arXiv:1806.01838`。通过对 block encoded matrix 应用投影器控制相位偏移，实现对矩阵奇异值的多项式变换。

**参数**:
- `A` - 输入矩阵 `(n × m)`
- `angles` - 相位角列表，用于定义所需的多项式变换
- `wires` - 量子比特列表

**示例**:
```python
from pyvqnet.qnn.vqc import VQC_QSVT, QMachine
from pyvqnet.dtype import float_dtype_to_complex_dtype
from pyvqnet import QTensor
import numpy as np

A = QTensor([[0.1, 0.2], [0.3, 0.4]])
angles = QTensor([0.1, 0.2, 0.3])
qm = QMachine(4, dtype=float_dtype_to_complex_dtype(A.dtype))
qm.reset_states(1)
qf = VQC_QSVT(A, angles, wires=[2, 1, 3])
z1 = qf(qm)
```

### VQC_QSVT_BlockEncoding

```python
from pyvqnet.qnn.vqc import VQC_QSVT_BlockEncoding

be = VQC_QSVT_BlockEncoding(A, wires)

# 前向传播
result = be(q_machine)
```

基于 VQC 的块编码，将矩阵 A 编码为更大酉矩阵的子块。是 QSVT 底层使用的块编码方案。

**参数**:
- `A` - 输入矩阵
- `wires` - 量子比特列表

**方法**:
- `adjoint()` - 返回伴随块编码矩阵
- `compute_matrix()` - 计算块编码矩阵

---

## QNG (Quantum Natural Gradient) 优化器

```python
from pyvqnet.qnn.vqc import QNG

qng = QNG(qmodel, stepsize=0.01, momentum=0)
```

量子自然梯度优化器，基于 `Quantum Natural Gradient <https://doi.org/10.22331/q-2020-05-25-269>`_。通过计算 Fubini-Study 度量张量的伪逆来实现在参数空间中沿最陡方向下降的自适应学习率。

**参数**:
- `qmodel` - 用户提供的 QModule 实例
- `stepsize` - 步长/学习率（默认 0.01）
- `momentum` - SGD 动量（默认 0）

**注意**:
1. 仅支持非批处理数据（batch_size=1）
2. 仅支持纯变分量子电路
3. 需要在模型 forward 上加 `@wrapper_calculate_qng` 装饰器

> **已知问题（2.18.2）**: 官方 QNG 示例在 2.18.2 下会抛出 `IndexError: list index out of range`（`qng.py::_calculate_metric_tensor` 内部）。原官方示例已移除；官方修复前建议使用 `QNSPSAOptimizer` 或常规优化器 + parameter-shift。

---

## QNSPSAOptimizer

```python
from pyvqnet.qnn.vqc import QNSPSAOptimizer

opt = QNSPSAOptimizer(
    stepsize=1e-3,
    regularization=1e-3,
    finite_diff_step=1e-2,
    resamplings=1,
    blocking=True,
    history_length=5,
    seed=None
)
```

量子自然 SPSA（Simultaneous Perturbation Stochastic Approximation）优化器。结合 SPSA 梯度估计与 Fubini-Study 度量张量信息的二阶随机优化器。

### 关键特性

- **梯度估计**: 使用对称扰动差分近似梯度
- **度量张量估计**: 通过四种电路评估的态重叠差异计算 Fubini-Study 度量
- **更新规则**: `x^{(t+1)} = x^{(t)} - η·ĝ⁻¹·∇̂f`

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `stepsize` | float | 1e-3 | 学习率 η |
| `regularization` | float | 1e-3 | 度量张量正则化 β（数值稳定性） |
| `finite_diff_step` | float | 1e-2 | 有限差分步长 ε |
| `resamplings` | int | 1 | 每次参数更新的采样平均次数 |
| `blocking` | bool | True | 仅接受不使损失增大的更新（加速收敛） |
| `history_length` | int | 5 | 用于设置 blocking 容忍度的历史损失长度 |
| `seed` | Optional[int] | None | 随机采样种子 |

### 方法

- `step(qmodel, *args, **kwargs)` - 执行一步优化，返回更新后的参数
- `step_and_cost(cost, *args, **kwargs)` - 执行一步优化并返回更新后的参数和损失值

**注意**:
1. 仅支持单样本（batch_size=1）
2. 仅支持 `MeasureAll` 作为测量方式
3. 输入参数需设置 `requires_grad = True`

**示例**:
```python
from pyvqnet.qnn.vqc import QNSPSAOptimizer, QMachine, rx, MeasureAll
from pyvqnet.tensor import QTensor
from pyvqnet.qnn.vqc import QModule

num_qubits = 2

class QModuleDemo(QModule):
    def __init__(self, name=""):
        super().__init__(name)
        self.qm = QMachine(num_qubits)
        self.ma = MeasureAll({"Z1 Z0": 1})

    def forward(self, params):
        qm = self.qm
        qm.reset_states(1)
        rx(qm, 0, params[0])
        rx(qm, 1, params[1])
        return self.ma(qm)

qmd = QModuleDemo()
params = QTensor([0.37454012, 0.95071431])
params.requires_grad = True

opt = QNSPSAOptimizer(stepsize=5e-2, seed=1)
for i in range(51):
    params = opt.step(qmd, params)
    loss = qmd(params)
    if i % 10 == 0:
        print(f"Step {i}: cost = {loss}")
```

---

## 量子算术操作

### QAdder (整数加法)

```python
from pyvqnet.qnn.vqc import QAdder

QAdder(qm, m, k, adder1, adder2, c, is_carry)
```

基于 MAJ（Majority）和 UMA（UnMajority and Add）门实现两个正整数的加法。结果存储在 `adder1` 中，进位存储在 `is_carry` 中。

**参数**:
- `qm` - QMachine 实例
- `m` - 被加数（整数）
- `k` - 加数（整数）
- `adder1` - 编码 m 的量子比特索引列表
- `adder2` - 编码 k 的量子比特索引列表
- `c` - 辅助量子比特索引（int 或 list）
- `is_carry` - 存储进位的量子比特索引（int 或 list）

**限制**: `batch_size=1`，仅支持非负整数。

**示例**:
```python
from pyvqnet.qnn.vqc import QAdder, QMachine, Samples

# 对 2 + 3 进行量子加法
dev = QMachine(8)
# adder1=2 (qubits 0,1), adder2=3 (qubits 2,3), c=4, is_carry=5
QAdder(dev, 2, 3, [0, 1], [2, 3], 4, 5)
ma = Samples(wires=[0, 1, 2])
y = ma(q_machine=dev)
```

### QMultiplier (整数乘法)

```python
from pyvqnet.qnn.vqc import QMultiplier

QMultiplier(qm, lhs, rhs, a, b, k, d)
```

基于 MAJ/UMA 门实现两个正整数乘法，结果存储在 `d` 中。

**参数**:
- `qm` - QMachine 实例
- `lhs` - 乘数1（整数）
- `rhs` - 乘数2（整数）
- `a` - 编码 lhs 的量子比特索引列表
- `b` - 编码 rhs 的量子比特索引列表
- `k` - 辅助量子比特索引列表（长度应 ≥ len(a)+1）
- `d` - 存储结果的量子比特索引列表（长度应 ≥ len(a)*2）

**示例**:
```python
from pyvqnet.qnn.vqc import QMultiplier, QMachine, Samples

# 2 × 3 = 6
wires_m = [0, 1]
wires_k = [2, 3]
wires_s = [4, 5, 6]
wires_c = [7, 8, 9, 10]
dev = QMachine(len(wires_m) + len(wires_k) + len(wires_c) + len(wires_s))

QMultiplier(dev, 2, 3, wires_m, wires_k, wires_s, wires_c)
m = Samples(wires=wires_c)
y = m(q_machine=dev)
```

### vqc_qft_add_to_register

```python
from pyvqnet.qnn.vqc import vqc_qft_add_to_register

vqc_qft_add_to_register(q_machine, m, k)
```

基于 QFT 的量子加法器，将一个数值 k 加到一个已编码在寄存器中的整数 m 上。操作流程：QFT → 相位旋转 → QFT⁻¹。

### vqc_qft_add_two_register

```python
from pyvqnet.qnn.vqc import vqc_qft_add_two_register

vqc_qft_add_two_register(q_machine, m, k, wires_m, wires_k, wires_solution)
```

将两个不同寄存器中的整数相加：|m⟩|k⟩|0⟩ → |m⟩|k⟩|m+k⟩。

### vqc_qft_mul

```python
from pyvqnet.qnn.vqc import vqc_qft_mul

vqc_qft_mul(q_machine, m, k, wires_m, wires_k, wires_solution)
```

基于 QFT 实现乘法：|m⟩|k⟩|0⟩ → |m⟩|k⟩|m·k⟩。

**参数**:
- `q_machine` - QMachine 实例
- `m` - 第一个整数
- `k` - 第二个整数
- `wires_m` - 编码 m 的量子比特索引
- `wires_k` - 编码 k 的量子比特索引
- `wires_solution` - 存储结果的量子比特索引

**示例**:
```python
from pyvqnet.qnn.vqc import (
    QMachine, Samples,
    vqc_qft_add_to_register,
    vqc_qft_add_two_register,
    vqc_qft_mul
)
import numpy as np

# vqc_qft_add_to_register: 3 + 7 = 10
dev = QMachine(4)
vqc_qft_add_to_register(dev, 3, 7)
ma = Samples()
y = ma(q_machine=dev)

# vqc_qft_mul: 3 × 7 = 21
wires_m = [0, 1, 2]
wires_k = [3, 4, 5]
wires_solution = [6, 7, 8, 9, 10]
dev = QMachine(len(wires_m) + len(wires_k) + len(wires_solution))
vqc_qft_mul(dev, 3, 7, wires_m, wires_k, wires_solution)
ma = Samples(wires=wires_solution)
y = ma(q_machine=dev)
```

---

## 张量网络（Tensor Network）后端

VQNet 2.18 提供了张量网络（TN）后端作为状态向量（state-vector）后端的替代方案。TN 后端使用张量网络收缩而非全状态向量模拟，在处理大规模量子电路时可大幅降低内存消耗。

### 何时使用 TN 后端

| 特性 | 状态向量（SV）后端 | 张量网络（TN）后端 |
|------|-------------------|-------------------|
| 内存 | O(2ⁿ) 指数增长 | O(poly(n)) 多项式增长 |
| 速度 | 密集模拟快 | 取决于电路结构（浅电路快） |
| 适用场景 | < 30 量子比特 | > 30 量子比特 / 浅电路 |
| MPS 支持 | 否 | 是（近似模拟） |
| 后端设置 | `pyvqnet`（默认） | `pyvqnet.backends.set_backend("pyvqnet")` |

### TNQModule

```python
TNQModule(use_jit=True, vectorized_argnums: Union[int, Sequence[int]]=0, name="", mesh=None, cotengra_options=None, tree_path=None)
```

在 `pyvqnet` 后端（默认后端）下，定义张量网络量子变分线路模型时应继承的基类。

> **警告**: `TNQModule` 仅支持量子类型的子模块（`TNQMachine`、量子门、ansatz 模板、测量模块与编码模块）。经典神经网络层（例如 `Linear`）不可作为 `TNQModule` 的子模块：JIT 编译的对象是纯量子线路，其中的经典计算将被忽略，或在编译时被固定为常量。构建经典-量子混合模型时，应将经典层定义在外层普通 `pyvqnet.nn.Module` 中，并将 `TNQModule` 作为其子模块；如需不经 JIT 编译执行，可直接使用 `TNQMachine`。

**参数**:
- `use_jit` - 是否开启即时编译，默认 True（native 与 torch TN 后端均默认为 True）
- `vectorized_argnums` - 要被向量化的参数，这些参数应该在同一维共享相同的批次形状，默认 0
- `name` - 模块名
- `mesh` - 用于分布式收缩的 `jax.sharding.Mesh`。为 None 时使用单设备 JIT；指定 mesh 时通过 SPMD 实现模型并行
- `cotengra_options` - 可选的字典，在指定 `mesh` 时传递给 `cotengra.ReusableHyperOptimizer`
- `tree_path` - 可选的预计算收缩路径文件（`.pkl`），指定后将跳过运行时的 cotengra 搜索

**注意**: 该类及其派生类仅适用于 `pyvqnet.backends.set_backend("pyvqnet")`。

### TNQMachine

```python
TNQMachine(num_wires, dtype=pyvqnet.kcomplex64, use_mps=False)
```

**参数**:
- `num_wires` - 量子比特数
- `dtype` - 数据类型（默认 kcomplex64）
- `use_mps` - 是否启用矩阵乘积态（MPS）近似（默认 False），适用于大量量子比特

**示例**:
```python
import pyvqnet
pyvqnet.backends.set_backend("pyvqnet")
from pyvqnet.qnn.vqc.tn.native import (
    TNQModule, TNQMachine,
    RX, RY, CNOT, PauliX, PauliZ,
    qmeasure, qcircuit, VQC_RotCircuit,
)
from pyvqnet.tensor import QTensor
# 或使用 Torch 版本:
# from pyvqnet.qnn.vqc.tn.torch import TNQModule, TNQMachine

class QModel(TNQModule):
    def __init__(self, batch_size=2):
        super().__init__()
        self.qm = TNQMachine(4)
        self.rx = RX(trainable=True, wires=0)
        self.ry = RY(trainable=True, wires=1)
        self.cnot = CNOT(wires=[0, 1])
        self.batch_size = batch_size

    def forward(self, x):
        # 基于 vmap，forward 内 x 已降一维，批次大小需显式指定
        self.qm.reset_states(self.batch_size)
        qcircuit.rx(q_machine=self.qm, wires=0, params=x[0])
        qcircuit.ry(q_machine=self.qm, wires=1, params=x[1])
        self.rx(q_machine=self.qm)
        self.ry(q_machine=self.qm)
        self.cnot(q_machine=self.qm)
        return qmeasure.Probability(wires=[0])(q_machine=self.qm)

x = QTensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]])
y = QModel(batch_size=2)(x)
y.backward(pyvqnet.tensor.ones_like(y))
print(y)
```

### TN native 量子门

`pyvqnet.qnn.vqc.tn.native` 与 `pyvqnet.qnn.vqc.tn.torch` 共享统一签名 `X(trainable=False, init_params=None, wires=None, dtype=pyvqnet.kcomplex64, use_dagger=False)`，支持上文门类列表中除 `X1`、`MultiControlledX` 外的全部门。

### TN native 测量接口

```python
VQC_Purity(qubits_idx, num_wires, qmachine)
VQC_DensityMatrixFromQstate(indices, num_wires, qmachine)
Probability(wires, name="")
MeasureAll(obs, name="")
HermitianExpval(obs, name="")
Samples(wires=None, obs=None, shots=1, name="")
```

- `VQC_Purity` 的 `qmachine` 为 `TNQMachine` 或 `TNQModule` 实例
- `Probability` 的 `wires` 以及 `MeasureAll` / `HermitianExpval` 的 `obs` 均为必填参数
- TN native 中不提供 `VQC_VarMeasure`；方差测量使用主模块的 `VQC_VarMeasure(q_machine, obs)`

---

## 新增测量类型

### MeasureAll

```python
MeasureAll(obs, name="")
```

```python
from pyvqnet.qnn.vqc import MeasureAll

measure = MeasureAll(obs={"Z0 Z1": 1, "X0": 0.5})
```

适用于需要直接指定可观测量 Pauli 字符串的场景。方便地构造泡利算符乘积的期望值测量。

**示例**:
```python
from pyvqnet.qnn.vqc import MeasureAll, QMachine, RX, RY
from pyvqnet.tensor import QTensor

qm = QMachine(2)
qm.reset_states(1)
rx = RX(trainable=False, wires=0)
ry = RY(trainable=False, wires=1)
rx(params=QTensor([0.5]), q_machine=qm)
ry(params=QTensor([0.3]), q_machine=qm)
# 测量 Z0⊗Z1 的期望值
measure = MeasureAll(obs={"Z0 Z1": 1})
result = measure(q_machine=qm)
```

### Samples

```python
Samples(wires=None, obs=None, shots=1, name="")
```

```python
from pyvqnet.qnn.vqc import Samples

sampler = Samples(wires=None)
y = sampler(q_machine=qm)
```

基于电路输出概率进行采样。若不指定 wires，则对所有量子比特采样。

### HermitianExpval

```python
HermitianExpval(obs, name="")
```

计算量子线路某个厄密特量的期望。`obs` 为必填参数，格式示例：`{'wires': (1, 0), 'observables': tensor.to_tensor(H)}`。

### VQC_VarMeasure

```python
VQC_VarMeasure(q_machine, obs)
```

提供的可观察量 `obs` 的方差。当前支持 `Hadamard`、`I`、`PauliX`、`PauliY`、`PauliZ` 几种 Observable。测量结果一般为 `[b, 1]`。

### VQC_Purity（状态向量后端）

```python
VQC_Purity(state, qubits_idx, num_wires)
```

从态矢中计算特定量子比特 `qubits_idx` 上的纯度 γ = Tr(ρ²)。注意张量网络后端的签名不同：`pyvqnet.qnn.vqc.tn.native.VQC_Purity(qubits_idx, num_wires, qmachine)`。

### VQC_DensityMatrixFromQstate（状态向量后端）

```python
VQC_DensityMatrixFromQstate(state, indices)
```

计算量子态在一组特定量子比特上的密度矩阵，返回大小 `(b, 2^len(indices), 2^len(indices))` 的密度矩阵。

---

## 量子机器学习模型接口

### Quanvolution

```python
Quanvolution(params_shape, strides=(1, 1), kernel_initializer=quantum_uniform, machine_type="cpu")
```

基于《Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits》实现的量子卷积，用变分量子电路替换经典卷积滤波器。

**参数**:
- `params_shape` - 参数的形状，应为二维
- `strides` - 切片窗口的步长，默认 `(1, 1)`
- `kernel_initializer` - 参数的卷积核初始化器
- `machine_type` - 机器类型字符串，默认 `"cpu"`

**示例**:
```python
from pyvqnet.qnn.qcnn import Quanvolution
import pyvqnet.tensor as tensor

qlayer = Quanvolution([4, 2], (3, 3))
x = tensor.arange(1, 25 * 25 * 3 + 1).reshape([3, 1, 25, 25])
y = qlayer(x)
print(y.shape)  # [3, 4, 8, 8]
y.backward()
```

### QGRU

```python
QGRU(para_num, num_of_qubits=4, input_size=100, hidden_size=100, batch_first=True)
```

基于变分量子线路实现的 GRU（门控循环单元），通过量子电路进行状态更新和记忆保持。

**参数**:
- `para_num` - 量子电路中的参数数量
- `num_of_qubits` - 量子比特的数量
- `input_size` - 输入数据的特征维度
- `hidden_size` - 隐藏单元的维度
- `batch_first` - 输入的第一维是否为批次数量

**示例**:
```python
import numpy as np
from pyvqnet.tensor import tensor
from pyvqnet.qnn.qgru import QGRU
from pyvqnet.dtype import kfloat32

para_num = 8
qgru = QGRU(para_num, 8, 4, 4, batch_first=True)
x = tensor.QTensor(np.random.randn(1, 1, 4), dtype=kfloat32)
output, h_t = qgru(x)
output.backward()
print("Output shape:", output.shape)
print("h_t shape:", h_t.shape)
```

### QLSTM

```python
QLSTM(para_num, num_of_qubits=4, input_size=100, hidden_size=100, batch_first=True)
```

结合量子计算与经典 LSTM 的混合模型，利用量子计算的并行性和经典 LSTM 的记忆能力处理序列数据。

**参数**:
- `para_num` - 量子电路中的参数数量
- `num_of_qubits` - 量子比特的数量
- `input_size` - 输入数据的特征维度
- `hidden_size` - 隐藏单元的维度
- `batch_first` - 输入的第一维是否为批次数量

**示例**:
```python
import numpy as np
from pyvqnet.tensor import tensor
from pyvqnet.qnn.qlstm import QLSTM
from pyvqnet.dtype import kfloat32

qlstm = QLSTM(4, 4, 4, 20, batch_first=True)
x = tensor.QTensor(np.random.randn(3, 5, 4), dtype=kfloat32)
output, (h_t, c_t) = qlstm(x)
print("Output shape:", output.shape)
print("h_t shape:", h_t.shape)
print("c_t shape:", c_t.shape)
```

### QRLModel

```python
QRLModel(num_qubits, n_layers=2)
```

使用变分量子电路的量子深度强化学习模型。

**参数**:
- `num_qubits` - 量子电路中所使用的量子比特数量
- `n_layers` - 变分量子电路中的层数

**示例**:
```python
from pyvqnet.tensor import QTensor
from pyvqnet.qnn.qrl import QRLModel

model = QRLModel(num_qubits=4, n_layers=2)
x = QTensor([[1.1, 0.3, 1.2, 0.6], [0.2, 1.1, 0, 1.1], [1.3, 1.3, 0.3, 0.3]])
output = model(x)
output.backward()
print("Model output:", output)
```

---

## 其他 ansatz 模板

除 `VQC_HardwareEfficientAnsatz` 外，VQNet 还提供了丰富的 **ExpressiveEntanglingAnsatz** 系列模板（`ExpressiveEntanglingAnsatz` ~ `ExpressiveEntanglingAnsatz_19`），提供了不同纠缠结构和旋转门组合的变分量子电路模板。可通过导入 `pyvqnet.qnn.vqc.sv.native.qcircuit` 使用。

---

## 电路编译优化

VQNet 提供了量子电路编译优化工具：

```python
from pyvqnet.qnn.vqc import (
    commute_controlled,
    merge_rotations,
    single_qubit_ops_fuse,
    wrapper_compile,
)
```

- `commute_controlled` - 将对换门交换以优化电路
- `merge_rotations` - 合并连续的旋转门
- `single_qubit_ops_fuse` - 融合单量子比特门
- `wrapper_compile` - 高阶编译包装函数

---

**Version**: VQNet 2.18.2
