---
name: vqnet2_api
description: PyVQNet/VQNet2.18.1 API 文档助手 - 量子机器学习编程专家。当用户需要"用 VQNet 2.18.1 写量子 ML 代码、实现 VQC、混合量子经典模型、GPU 训练、分布式训练"等任务时，自动调用此 Skill。触发关键词: VQNet, pyvqnet, QTensor, QuantumLayer, 量子神经网络, 变分量子线路, 量子机器学习, QVC, VSQL, Quanvolution。
license: Apache License 2.0
---

# VQNet 2.18.1 分层 Skill 体系

**重要规则**: 本 Skill 所有示例代码均来自官方文档。AI 必须先读取对应的 `references/*.md` 文件获取准确示例，**禁止凭记忆编造 VQNet 代码**。

---

## 🔧 基础层 Skills

### 1. 环境配置 Skill (env_setup)

**能力**: 安装 VQNet、配置 Python 环境、选择 CPU/GPU 后端、验证安装。

**关键 API**:
- `pip install pyvqnet --upgrade`
- `pyvqnet.DEV_CPU`, `pyvqnet.DEV_GPU_0`
- `QTensor.toGPU()`, `Module.toGPU()`

**参考文件**: `references/install_env.md`

**最小示例**:
```python
# 安装验证
import pyvqnet
from pyvqnet.tensor import QTensor, ones
a = ones([4, 5])
print(a)

# GPU 测试
from pyvqnet import DEV_GPU_0
a = ones([4, 5], device=DEV_GPU_0)
print(a)
```

---

### 2. QTensor 基础 Skill (qtensor_basic)

**能力**: 创建张量、数学运算、形状操作、自动微分。

**关键 API**:
- `QTensor(data, requires_grad=False, device=DEV_CPU, dtype=None)`
- `ones`, `zeros`, `arange`, `randn`, `randu`
- `reshape`, `transpose`, `permute`, `flatten`
- `backward()`, `zero_grad()`, `grad`
- `to_numpy()`, `item()`

**参考文件**: `references/qtensor.md`

**最小示例**:
```python
from pyvqnet.tensor import QTensor, arange

# 创建带梯度的张量
a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
print(a.shape)  # [4]
print(a.ndim)   # 1
print(a.size)   # 4

# 自动微分
b = 2 * a + 3
b.backward()
print(a.grad)  # [2., 2., 2., 2.]

# 形状操作
c = arange(1, 25).reshape([2, 3, 4])
print(c.shape)  # [2, 3, 4]
```

**数据类型**:
| dtype | 描述 |
|-------|------|
| `pyvqnet.kfloat32` | 32位浮点（默认） |
| `pyvqnet.kfloat64` | 64位双精度 |
| `pyvqnet.kint64` | 64位整数（标签用） |
| `pyvqnet.kcomplex64` | 64位复数 |
| `pyvqnet.kbool` | 布尔 |

---

### 3. 经典神经网络 Skill (classical_nn)

**能力**: 构建 Module、定义层、损失函数、优化器、GPU训练。

**关键 API**:
- `pyvqnet.nn.module.Module` - 模型基类
- `pyvqnet.nn.Linear`, `Conv2D`, `BatchNorm2D`
- `pyvqnet.nn.loss.MSELoss`, `CrossEntropyLoss`, `CategoricalCrossEntropy`
- `pyvqnet.optim.SGD`, `Adam`, `AdamW`
- `Module.toGPU(device)`, `QTensor.toGPU()`

**参考文件**: `references/classic_nn.md`

**最小示例**:
```python
from pyvqnet.nn import Module, Linear, ReLU, Sequential
from pyvqnet.nn.loss import MSELoss
from pyvqnet.optim import Adam
from pyvqnet.tensor import QTensor, randn

class MLP(Module):
    def __init__(self):
        super().__init__()
        self.net = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MLP()
optimizer = Adam(model.parameters(), lr=1e-3)
loss_fn = MSELoss()

# 训练循环
x = randn([32, 4])
y = randn([32, 1])

pred = model(x)
loss = loss_fn(y, pred)  # 注意: (标签, 预测值)
optimizer.zero_grad()
loss.backward()
optimizer._step()
```

---

### 4. 神经网络工具 Skill (nn_utils)

**能力**: 神经网络工具 (random seed, parameter initializers).

**关键 API**:
- `pyvqnet.utils.set_random_seed(seed)` - 设置全局随机种子
- `pyvqnet.utils.get_random_seed()` - 获取当前随机种子
- `pyvqnet.utils.initializer.he_normal` - He 正态初始化
- `pyvqnet.utils.initializer.he_uniform` - He 均匀初始化
- `pyvqnet.utils.initializer.xavier_normal` - Xavier 正态初始化
- `pyvqnet.utils.initializer.xavier_uniform` - Xavier 均匀初始化
- `pyvqnet.utils.initializer.quantum_uniform` - 量子参数均匀初始化
- `pyvqnet.utils.initializer.uniform` - 均匀分布初始化
- `pyvqnet.utils.initializer.normal` - 正态分布初始化

**参考文件**: `references/utils.md`

**最小示例**:
```python
from pyvqnet.utils import set_random_seed, get_random_seed
from pyvqnet.utils.initializer import he_normal, xavier_uniform
from pyvqnet.nn.parameter import Parameter

# 设置随机种子
set_random_seed(256)
print(get_random_seed())  # 256

# 参数初始化器
param = Parameter(shape=[2, 3], initializer=he_normal)
print(param)
```

---

## ⚛️ 量子计算层 Skills

### 5. 量子电路基础 Skill (qcircuit_basic)

**能力**: 使用 pyqpanda3 构建 QCircuit、量子门操作、测量。

**关键 API** (pyqpanda3):
- `pq.CPUQVM()` - 模拟器
- `pq.QCircuit()` - 量子电路
- `pq.QProg()` - 量子程序
- `pq.H`, `pq.X`, `pq.RX`, `pq.RY`, `pq.RZ`, `pq.CNOT`
- `pq.measure(qubit, cbit)`

**参考文件**: `references/quantum_layers.md`

**最小示例**:
```python
import pyqpanda3.core as pq

machine = pq.CPUQVM()
qubits = range(4)
circuit = pq.QCircuit()

# 添加量子门
circuit << pq.H(0) << pq.H(1) << pq.H(2) << pq.H(3)
circuit << pq.CNOT(0, 1) << pq.RZ(1, 0.5)

prog = pq.QProg()
prog << circuit

# 测量
for i, q in enumerate(qubits[:2]):
    prog << pq.measure(q, i)
```

---

### 6. QuantumLayer Skill (quantum_layer)

**能力**: 将量子电路封装为可微分层，支持 parameter-shift 梯度计算。

**关键 API**:
- `QuantumLayer(qprog_with_measure, para_num)` - pyqpanda3 版本
- `QpandaQProgVQCLayer(origin_qprog_func, para_num, qvm_type="cpu")`
- `QuantumBatchAsyncQcloudLayer(...)` - 真机运行
- `QuantumLayerAdjoint(...)` - adjoint 梯度方法

**关键签名**: `qprog_with_measure(input, param)` - **input 在前，param 在后**

**参考文件**: `references/quantum_layers.md`

**最小示例**:
```python
from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
from pyvqnet.qnn.pq3.measure import ProbsMeasure
from pyvqnet.tensor import QTensor
import pyqpanda3.core as pq

def circuit(input, param):
    machine = pq.CPUQVM()
    qubits = range(4)
    cir = pq.QCircuit()

    # 编码输入
    for i in range(4):
        cir << pq.H(i) << pq.RZ(i, input[i])

    # 变分部分
    cir << pq.CNOT(0, 1) << pq.RZ(1, param[0])
    cir << pq.CNOT(1, 2) << pq.RZ(2, param[1])

    prog = pq.QProg()
    prog << cir
    return ProbsMeasure(machine, prog, [0, 1, 2, 3])

# 3 个可训练参数
layer = QuantumLayer(circuit, 3)

x = QTensor([[0.1, 0.2, 0.3, 0.4]])
y = layer(x)
y.backward()
print(layer.m_para.grad)
```

---

### 7. VQC 自动微分 Skill (vqc_native_autodiff)

**能力**: 使用 VQNet 内置的 VQC 模块，不依赖 pyqpanda，自动微分模拟。

**关键 API**:
- `pyvqnet.qnn.vqc.QMachine(num_wires)` - 模拟器
- `pyvqnet.qnn.vqc.Hadamard`, `RX`, `RY`, `RZ`, `CNOT`
- `pyvqnet.qnn.vqc.Probability(wires)` - 概率测量
- `VQC_HardwareEfficientAnsatz` - 硬件高效 ansatz

**关键**: 必须在 forward 开头调用 `device.reset_states(batchsize)`

**参考文件**: `references/vqc.md`

**最小示例**:
```python
from pyvqnet.nn import Module, Linear
from pyvqnet.qnn.vqc import QMachine, RZ, Probability
from pyvqnet.qnn.vqc import VQC_HardwareEfficientAnsatz
from pyvqnet import tensor

class QModel(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(4, 2)
        self.encode = RZ(wires=0)
        self.ansatz = VQC_HardwareEfficientAnsatz(4, ["rx", "RY", "rz"],
                                                   entangle_gate="cnot",
                                                   depth=2)
        self.measure = Probability(wires=[0, 2])
        self.device = QMachine(4)

    def forward(self, x):
        # 必须重置状态
        self.device.reset_states(x.shape[0])

        y = self.linear(x)
        self.encode(params=y[:, 0], q_machine=self.device)
        self.ansatz(q_machine=self.device)
        return self.measure(q_machine=self.device)

model = QModel()
x = tensor.arange(1.0, 13).reshape([3, 4])
x.requires_grad = True
y = model(x)
y.backward()
```

---

### 8. 量子测量与熵 Skill (quantum_measurements)

**能力**: 量子测量与熵函数 (expval, QuantumMeasure, ProbsMeasure, DensityMatrix, VN_Entropy, Mutual_Info, Purity).

**关键 API**:
- `pyvqnet.qnn.pq3.measure.expval(machine, prog, pauli_str_dict)` - Pauli 字符串期望值
- `pyvqnet.qnn.pq3.measure.QuantumMeasure(machine, prog, measure_qubits, shots)` - 量子测量 (Monte Carlo)
- `pyvqnet.qnn.pq3.measure.ProbsMeasure(machine, prog, measure_qubits, shots)` - 概率测量
- `pyvqnet.qnn.pq3.measure.DensityMatrixFromQstate(state, indices)` - 约化密度矩阵
- `pyvqnet.qnn.pq3.measure.VN_Entropy(state, indices, base)` - Von Neumann 熵
- `pyvqnet.qnn.pq3.measure.Mutal_Info(state, indices0, indices1, base)` - 互信息
- `pyvqnet.qnn.pq3.measure.Purity(state, qubits_idx)` - 纯度

**参考文件**: `references/measurement.md`

**最小示例**:
```python
from pyvqnet.qnn.pq3.measure import expval, VN_Entropy, Mutal_Info, Purity
import pyqpanda3.core as pq

# 期望值计算
machine = pq.CPUQVM()
cir = pq.QCircuit(3)
cir << pq.H(0) << pq.CNOT(0, 1) << pq.RY(1, 0.5)
prog = pq.QProg(cir)
exp2 = expval(machine, prog, {'Z0 X1': 10, 'Y2': -0.543})
print(exp2)

# 熵与互信息 (基于完整量子态)
qstate = [(0.902 + 0j), -0.067j, (0.183 + 0j), -0.329j,
          (0.037 + 0j), -0.067j, (0.183 + 0j), -0.014j]
print(VN_Entropy(qstate, [0, 1]))   # Von Neumann 熵
print(Mutal_Info(qstate, [0], [2])) # 互信息
print(Purity(qstate, [1]))          # 纯度
```

---

### 9. QNN 架构总览 Skill (qnn_overview)

**能力**: QNN 架构总览与最佳实践 (hybrid model patterns, component selection, common issues).

**关键 API**:
- `QuantumLayer` - 通用变分量⼦层
- `QpandaQProgVQCLayer` - QProg 返回型变分层
- `QuantumLayerAdjoint` - adjoint 梯度 VQE
- `QLinear` - 量子全连接层 (`pyvqnet.qnn.qlinear`)
- `QConv` - 量子卷积层 (`pyvqnet.qnn.qcnn.qconv`)
- 嵌入模板: `AmplitudeEmbeddingCircuit`, `AngleEmbeddingCircuit`, `IQPEmbeddingCircuits`
- Ansatz 模板: `HardwareEfficientAnsatz`, `BasicEntanglerTemplate`

**参考文件**: `references/qnn.md`

**最佳实践**:
| 数据维度 | 编码方式 | 所需量子比特 |
|----------|----------|-------------|
| N 特征 | Angle 编码 | N 个 |
| 2^N 特征 | Amplitude 编码 | N 个 |
| 任意 N ≤ 量子比特 | IQP 编码 | N 个 |

**最小示例**:
```python
import pyqpanda3.core as pq
from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
from pyvqnet.qnn.pq3.measure import ProbsMeasure
from pyvqnet.nn import Sequential, Linear, ReLU

def quantum_forward(x, params):
    n_qubits = 4
    machine = pq.CPUQVM()
    cir = pq.QCircuit()
    for i in range(n_qubits):
        if i < len(x):
            cir << pq.H(i) << pq.RY(i, x[i])
    for i in range(n_qubits - 1):
        cir << pq.CNOT(i, i + 1) << pq.RX(i + 1, params[i])
    prog = pq.QProg() << cir
    return ProbsMeasure(machine, prog, range(n_qubits))

# 混合模型
model = Sequential(
    Linear(784, 16), ReLU(),
    Linear(16, 4),
    QuantumLayer(quantum_forward, 3),  # 量子层
    Linear(4, 10)                       # 经典输出层
)
```

---

### 10. 量子电路模板与拟设 Skill (circuit_templates)

**能力**: 量子电路模板与拟设 (Embedding circuits, ansatz templates, fermionic operators, UCCSD).

**关键 API**:
- `AmplitudeEmbeddingCircuit` - 振幅嵌入 (需 L2 归一化)
- `AngleEmbeddingCircuit` - 角度嵌入 (rotation='X'/'Y'/'Z')
- `IQPEmbeddingCircuits` - IQP 嵌入 (Havlicek et al. 2018)
- `HardwareEfficientAnsatz` - 硬件高效 ansatz (`create_ansatz(params)`)
- `StronglyEntanglingTemplate` - 强纠缠模板 (arXiv:1804.00633)
- `BasicEntanglerTemplate` - 基础纠缠模板
- `UCCSD` - 单双激发幺正耦合簇
- `FermionicSingleExcitation` / `FermionicDoubleExcitation` - 费米子激发算符

**参考文件**: `references/quantum_templates.md`

**最小示例**:
```python
import numpy as np
import pyqpanda3.core as pq
from pyvqnet.qnn.pq3.template import (
    AmplitudeEmbeddingCircuit,
    AngleEmbeddingCircuit,
    IQPEmbeddingCircuits
)
from pyvqnet.qnn.pq3.ansatz import HardwareEfficientAnsatz
from pyvqnet.qnn.pq3 import UCCSD
from pyvqnet.tensor import tensor

# 振幅嵌入 (必须归一化)
feat = np.array([2.2, 1.0, 4.5, 3.7]) / np.linalg.norm([2.2, 1.0, 4.5, 3.7])
cir = AmplitudeEmbeddingCircuit(feat, range(2))

# 硬件高效 ansatz
ansatz = HardwareEfficientAnsatz(range(4), ["rx", "RY", "rz"],
                                  entangle_gate="cnot", depth=1)
w = tensor.ones([ansatz.get_para_num()])
cir2 = ansatz.create_ansatz(w)

# UCCSD (量子化学)
weight = tensor.zeros([8])
cir3 = UCCSD(weight,
             wires=[0, 1, 2, 3, 4, 5],
             s_wires=[[0, 1, 2], [0, 1, 2, 3, 4]],
             d_wires=[[[0, 1], [2, 3]], [[0, 1], [2, 3, 4, 5]]],
             init_state=[1, 1, 0, 0, 0, 0],
             qubits=range(6))
```

---

## 🎯 场景任务层 Skills

### 11. QML 分类器 Skill (qml_classifier)

**能力**: 实现量子变分分类器 - QVC、VSQL、Circuit-centric classifier。

**参考文件**: `references/qml_demos.md`

**典型任务**: 二分类、多分类、MNIST 分类

**QVC 示例**:
```python
from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
from pyvqnet.nn.module import Module
from pyvqnet.optim import SGD
from pyvqnet.nn.loss import CategoricalCrossEntropy
import pyqpanda3.core as pq

def qvc_circuit(input, weights):
    machine = pq.CPUQVM()
    qubits = range(4)
    cir = pq.QCircuit()

    # 编码：二进制数据转为量子态
    for i in range(4):
        if input[i] == 1:
            cir << pq.X(i)

    # 变分层：Rot + CNOT
    weights = weights.reshape([2, 4, 3])
    for layer in range(2):
        for q in range(4):
            cir << pq.RZ(q, weights[layer, q, 0])
            cir << pq.RY(q, weights[layer, q, 1])
            cir << pq.RZ(q, weights[layer, q, 2])
        for i in range(3):
            cir << pq.CNOT(i, i+1)
        cir << pq.CNOT(3, 0)

    prog = pq.QProg()
    prog << cir
    # 返回概率测量
    ...

class QVCModel(Module):
    def __init__(self):
        super().__init__()
        self.qvc = QuantumLayer(qvc_circuit, 24)

    def forward(self, x):
        return self.qvc(x)
```

---

### 12. Quanvolution Skill (qml_quanvolution)

**能力**: 量子卷积层用于图像分类。

**关键 API**: 参考 `references/qml_demos.md`

---

### 13. 混合模型 Skill (qml_hybrid)

**能力**: CNN + QNN 混合模型、经典预处理 + 量子计算 + 经典后处理。

**参考文件**: `references/qml_demos.md`

---

## 🖥️ 系统能力层 Skills

### 14. 分布式训练 Skill (distributed)

**能力**: MPI 多进程训练、NCCL GPU 通信、多节点部署。

**关键 API**:
- `pyvqnet.distributed.CommController("mpi")` / `CommController("nccl")` - 通信控制器
- `vqnetrun -np N python train.py`

**参考文件**: `references/distributed.md`

**要求**: 仅支持 Linux；CPU 分布式需 mpich-mpicxx 4.1.2 + mpi4py；GPU 需 NCCL（随包附带）

---

### 15. 真机运行 Skill (real_chip)

**能力**: 提交量子电路到本源量子云真机运行。

**关键 API**:
- `QuantumBatchAsyncQcloudLayer(origin_qprog_func, qcloud_token, para_num)`
- Token 获取: https://qcloud.originqc.com.cn/

**参考文件**: `references/quantum_layers.md`

**示例**:
```python
import os
from pyvqnet.qnn.pq3.quantumlayer import QuantumBatchAsyncQcloudLayer

token = os.getenv("QCLOUD_TOKEN")  # 不要硬编码

def circuit(input, param):
    ...

layer = QuantumBatchAsyncQcloudLayer(
    circuit, token, 2,
    submit_kwargs={"test_qcloud_fake": True}  # 测试模式
)
```

---

### 16. 量子大模型微调 Skill (quantum_llm)

**能力**: 结合 Llama Factory 使用 VQC 进行大模型微调。

**关键 API**:
- `quantum-llm` 库 + `peft_vqc`
- `finetuning_type: vqc`

**参考文件**: `references/quantum_llm.md`；trl 损失函数（sft/dpo/ppo/grpo/reward）参考 `references/llm_ops.md`

---

### 17. PyTorch 后端 Skill (torch_backend)

**能力**: PyTorch 后端切换 (set_backend, get_backend, TorchModule, QModule).

**关键 API**:
- `pyvqnet.backends.set_backend(backend_name)` - 切换全局后端 ("pyvqnet", "pyvqnet-ad", "torch", "torch-native")
- `pyvqnet.backends.get_backend(t=None)` - 获取当前后端
- `pyvqnet.nn.torch.TorchModule` - PyTorch 后端经典神经网络基类
- `pyvqnet.qnn.vqc.torch.QModule` - PyTorch 后端 VQC 自动微分基类
- `QTensor.data` 变为 `torch.Tensor` (torch 后端下)

**注意事项**:
- 调用 `set_backend("torch")` 后，QTensor 底层变为 torch.Tensor
- 不同后端创建的 QTensor 不能混合使用
- 需要 PyTorch（需自行安装；GPU 场景建议与 CUDA 12.x 匹配，官方基准 torch 2.11.0+cu126 验证）
- VQNet 不自带 PyTorch，需用户自行安装

**参考文件**: `references/torch_api.md`

**最小示例**:
```python
import pyvqnet
import torch
from pyvqnet.tensor import QTensor

# 切换到 PyTorch 后端
pyvqnet.backends.set_backend("torch")

# 创建张量 - 底层是 torch.Tensor
t = QTensor([1.0, 2.0, 3.0], requires_grad=True)
print(type(t.data))  # <class 'torch.Tensor'>

# 计算与反向传播
result = t.sum()
result.backward()
print(t.grad)

# 查看当前后端
print(pyvqnet.backends.get_backend())  # "torch"
```

---

## ⚠️ Anti-Patterns & FAQ

### 常见错误

1. **参数顺序错误**: QuantumLayer 函数签名必须是 `(input, param)`，不是 `(param, input)`
2. **忘记 requires_grad**: 可训练参数必须设置 `requires_grad=True`
3. **忘记 reset_states**: VQC 模型必须在 forward 开头调用 `device.reset_states(batchsize)`
4. **混合后端**: 不同 backend 的 QTensor 不能混用
5. **ModuleList vs List**: 子模块列表要用 `ModuleList`，不能用 Python `list`
6. **dtype 错误**: Embedding 输入需 `kint64`，交叉熵标签需 `kint64`
7. **硬编码 Token**: QCloud Token 必须用环境变量
8. **AmplitudeEmbedding 未归一化**: 输入特征必须手动调用 `x / np.linalg.norm(x)` 归一化, L2 norm 必须为 1
9. **HardwareEfficientAnsatz 参数遗漏**: pyqpanda3 版本必须使用 `get_para_num()` + `create_ansatz(params)` 组合, 不能调用无参方法
10. **损失函数参数顺序**: VQNet 损失函数 `loss_fn(label, pred)` — 标签在前，预测值在后（与 PyTorch 相反）

### GPU 训练要点

- 数据和模型都要移动到 GPU: `QTensor.toGPU()`, `Module.toGPU()`
- CUDA 架构要求: sm_80 (A100) 或 sm_86 (RTX 30系列)

---

## 📚 参考文件索引

| 需求 | 读取文件 |
|------|----------|
| 安装/环境/FAQ | `references/install_env.md` |
| QTensor API | `references/qtensor.md` |
| 经典神经网络 | `references/classic_nn.md` |
| 神经网络工具 | `references/utils.md` |
| QuantumLayer | `references/quantum_layers.md` |
| VQC 自动微分 | `references/vqc.md` |
| 量子测量与熵 | `references/measurement.md` |
| QNN 架构总览 | `references/qnn.md` |
| 电路模板与拟设 | `references/quantum_templates.md` |
| QML Demo | `references/qml_demos.md` |
| 分布式训练 | `references/distributed.md` |
| 量子大模型 | `references/quantum_llm.md` |
| 大模型算子与 TRL 微调损失 | `references/llm_ops.md` |
| PyTorch 后端 / torch 量子线路（sv.torch、tn.torch） | `references/torch_api.md` |

---

## 🔄 执行流程

**Step 1**: 分析用户需求 → 确定 Skill 层级
**Step 2**: 读取对应的 `references/*.md` 文件
**Step 3**: 从参考文件中提取准确的示例代码
**Step 4**: 根据用户需求调整参数，但保持 API 签名正确
**Step 5**: 输出代码前验证导入路径、参数顺序

---

**Version**: 2.18.1
**Compatibility**: VQNet 2.18.1+ with pyqpanda3