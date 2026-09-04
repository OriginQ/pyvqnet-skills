# VQC 自动微分 API Reference

> **重要**: 所有示例代码均来自官方文档。VQC 模块不依赖 pyqpanda，使用 VQNet 内置的自动微分模拟。

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

# 参数化旋转门 - 设置 has_params=True, trainable=True
device = QMachine(4)

# 可训练的 RX 门
layer = RX(has_params=True, trainable=True, wires=0)
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
from pyvqnet.qnn.vqc import Probability, QMachine

# 概率测量
measure = Probability(wires=[0, 2])
device = QMachine(4)
device.reset_states(1)
prob = measure(q_machine=device)
print(prob)
```

---

## Ansatz 模板

### VQC_HardwareEfficientAnsatz

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
        self.encode1(params=y[:, 0], q_machine=self.device)
        self.encode2(params=y[:, 1], q_machine=self.device)
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
        # 设置 has_params=True, trainable=True 来添加可训练参数
        self.vqc = RZ(has_params=True, trainable=True, wires=1)
        self.measure = Probability(wires=[0, 2])
        self.device = QMachine(4)

    def forward(self, x):
        self.device.reset_states(x.shape[0])
        y = self.linearx(x)
        self.encode1(params=y[:, 0], q_machine=self.device)
        self.encode2(params=y[:, 1], q_machine=self.device)
        self.vqc(q_machine=self.device)  # 使用可训练参数
        self.ansatz(q_machine=self.device)
        return self.measure(q_machine=self.device)
```

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
3. **trainable**: 设置 `has_params=True, trainable=True` 添加可训练参数
4. **dtype**: 参数精度为 `kfloat32`，门内部为 `kcomplex64`

---

## 支持的量子门

| 门 | 函数 | 类 |
|---|---|---|
| I | `i()` | `I` |
| Hadamard | `hadamard()` | `Hadamard` |
| X | `paulix()` | `PauliX` |
| Y | `pauliy()` | `PauliY` |
| Z | `pauliz()` | `PauliZ` |
| RX | `rx()` | `RX` |
| RY | `ry()` | `RY` |
| RZ | `rz()` | `RZ` |
| CNOT | `cnot()` | `CNOT` |
| CZ | `cz()` | `CZ` |
| SWAP | `swap()` | `SWAP` |
| S | `s()` | `S` |
| T | `t()` | `T` |

---

## QuantumAdjointLayer (adjoint gradient)

```python
from pyvqnet.qnn.vqc import QuantumAdjointLayer, QuantumLayerAdjoint

adjoint_model = QuantumAdjointLayer(vqc_module, use_qpanda=False, name="")
# QuantumAdjointLayer and QuantumLayerAdjoint are the same class
```

记忆高效的梯度计算方法，基于伴随矩阵方法（adjoint method）。参考论文: `Efficient calculation of gradients in classical simulations of variational quantum algorithms <https://arxiv.org/abs/2009.02823>`_。

**参数**:
- `vqc_module` - VQC Module 实例（其 QMachine 必须设置 `grad_mode="adjoint"`）
- `use_qpanda` - 是否使用 pyqpanda 进行前向加速（默认 False）。复杂电路下 qpanda 有速度优势
- `name` - 模块名称

**注意**:
1. 仅支持单参数量子门的 VQC 模块
2. QMachine 必须设置 `grad_mode="adjoint"`
3. 支持输入数据梯度和可训练参数梯度

**示例**:
```python
from pyvqnet import tensor
from pyvqnet.qnn.vqc import (
    QuantumAdjointLayer, QMachine,
    RX, RY, RZ, CNOT, T,
    VQC_HardwareEfficientAnsatz, MeasureAll
)
import pyvqnet

class QModel(pyvqnet.nn.Module):
    def __init__(self, num_wires, dtype, grad_mode=""):
        super(QModel, self).__init__()
        self._num_wires = num_wires
        self.qm = QMachine(num_wires, dtype=dtype, grad_mode=grad_mode)
        self.rx_layer = RX(has_params=True, trainable=False, wires=0)
        self.ry_layer = RY(has_params=True, trainable=False, wires=1)
        self.rz_layer = RZ(has_params=True, trainable=False, wires=1)
        self.rz_layer2 = RZ(has_params=True, trainable=True, wires=1)
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
adjoint_model = QuantumAdjointLayer(qunatum_model)
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
- `serialize()` - 序列化电路图
- `print_contents()` - 打印电路内容

**示例**:
```python
from pyvqnet.qnn.vqc import CircuitGraph, QMachine, RX, RY, CNOT, MeasureAll

qm = QMachine(3)
qm.reset_states(1)
rx = RX(has_params=True, trainable=False, wires=0)
ry = RY(has_params=True, trainable=True, wires=1)
cnot = CNOT(wires=[0, 1])
measure = MeasureAll(obs={"Z0": 1, "Z1": 1})

rx(params=[0.5], q_machine=qm)
cnot(q_machine=qm)
ry(q_machine=qm)
result = measure(q_machine=qm)

cg = CircuitGraph(
    qm.op_history,
    range(qm.num_wires),
    qm.train_params_indices
)
print(f"Circuit depth: {cg.get_depth()}")
print(f"Circuit hash: {cg.hash}")
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

**示例**:
```python
from pyvqnet.qnn.vqc import (
    QNG, wrapper_calculate_qng,
    QMachine, RX, RY, RZ, CNOT,
    PauliX, MeasureAll
)
from pyvqnet.tensor import QTensor
import pyvqnet, numpy as np

class QModel(pyvqnet.nn.Module):
    def __init__(self, num_wires, dtype):
        super(QModel, self).__init__()
        self._num_wires = num_wires
        self._dtype = dtype
        self.qm = QMachine(num_wires, dtype=dtype)
        self.rz_layer1 = RZ(has_params=True, trainable=False, wires=0)
        self.rz_layer2 = RZ(has_params=True, trainable=False, wires=1)
        self.l_train1 = RY(has_params=True, trainable=True, wires=1)
        self.l_train2 = RX(has_params=True, trainable=True, wires=2)
        self.xlayer = PauliX(wires=0)
        self.cnot01 = CNOT(wires=[0, 1])
        self.cnot12 = CNOT(wires=[1, 2])
        self.measure = MeasureAll(obs={'Y0': 1})

    @wrapper_calculate_qng
    def forward(self, x, *args, **kwargs):
        self.qm.reset_states(x.shape[0])
        self.rz_layer1(q_machine=self.qm, params=x[:, [0]])
        self.rz_layer2(q_machine=self.qm, params=x[:, [1]])
        self.cnot01(q_machine=self.qm)
        self.cnot12(q_machine=self.qm)
        self.l_train1(q_machine=self.qm)
        self.l_train2(q_machine=self.qm)
        self.cnot01(q_machine=self.qm)
        self.cnot12(q_machine=self.qm)
        return self.measure(q_machine=self.qm)

qmodel = QModel(3, pyvqnet.kcomplex64)
x = QTensor([[1111.0, 2222, 444]], requires_grad=True)

qng = QNG(qmodel, 0.01)
qng.step(x)
qng.step(x)
```

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

### TNQMachine / TNQModule

```python
import pyvqnet
pyvqnet.backends.set_backend("pyvqnet")
from pyvqnet.qnn.vqc.tn.native import TNQModule, TNQMachine
# 或使用 Torch 版本:
# from pyvqnet.qnn.vqc.tn.torch import TNQModule, TNQMachine

machine = TNQMachine(num_wires, dtype=pyvqnet.kcomplex64, use_mps=False)
```

**参数**:
- `num_wires` - 量子比特数
- `dtype` - 数据类型（默认 kcomplex64）
- `use_mps` - 是否启用矩阵乘积态（MPS）近似（默认 False），适用于大量量子比特

**示例**:
```python
import pyvqnet
from pyvqnet.nn import Parameter
pyvqnet.backends.set_backend("pyvqnet")
from pyvqnet.qnn.vqc.tn.native import (
    TNQModule, TNQMachine,
    RX, RY, CNOT, PauliX, PauliZ,
    qmeasure, qcircuit, VQC_RotCircuit,
)

class QModel(TNQModule):
    def __init__(self):
        super().__init__()
        self.qm = TNQMachine(4)
        self.rx = RX(has_params=True, trainable=True, wires=0)
        self.ry = RY(has_params=True, trainable=True, wires=1)
        self.cnot = CNOT(wires=[0, 1])

    def forward(self, x):
        self.qm.reset_states(x.shape[0])
        self.rx(params=x[:, [0]], q_machine=self.qm)
        self.ry(params=x[:, [1]], q_machine=self.qm)
        self.cnot(q_machine=self.qm)
        return qmeasure.Probability(wires=[0])(q_machine=self.qm)
```

---

## 新增测量类型

### MeasureAll

```python
from pyvqnet.qnn.vqc import MeasureAll

measure = MeasureAll(obs={"Z0 Z1": 1, "X0": 0.5})
```

适用于需要直接指定可观测量 Pauli 字符串的场景。方便地构造泡利算符乘积的期望值测量。

**示例**:
```python
from pyvqnet.qnn.vqc import MeasureAll, QMachine, RX, RY

qm = QMachine(2)
qm.reset_states(1)
rx = RX(has_params=True, trainable=False, wires=0)
ry = RY(has_params=True, trainable=False, wires=1)
rx(params=[0.5], q_machine=qm)
ry(params=[0.3], q_machine=qm)
# 测量 Z0⊗Z1 的期望值
measure = MeasureAll(obs={"Z0 Z1": 1})
result = measure(q_machine=qm)
```

### Samples

```python
from pyvqnet.qnn.vqc import Samples

sampler = Samples(wires=None)
y = sampler(q_machine=qm)
```

基于电路输出概率进行采样。若不指定 wires，则对所有量子比特采样。

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

**Version**: VQNet 2.18.1
