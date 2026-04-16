# VQC 自动微分 API Reference

> 来源: VQNET2.0-tutorial/source/rst/vqc.rst
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
from pyvqnet.qnn.vqc.qcircuit import VQC_HardwareEfficientAnsatz

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
from pyvqnet.qnn.vqc.qcircuit import VQC_HardwareEfficientAnsatz, RZ
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
from pyvqnet.qnn.vqc.qcircuit import VQC_HardwareEfficientAnsatz, RZ
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

**Version**: VQNet 2.0
**Source**: VQNET2.0-tutorial/source/rst/vqc.rst