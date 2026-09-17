# Quantum Layers API Reference

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

**与 QuantumLayerV2 的区别**:

| 维度 | QuantumLayerV2 (`QuantumLayer`) | QuantumLayerV3 (`QpandaQProgVQCLayer`) |
|------|-------------------------------|----------------------------------------|
| 函数签名 | `(input, param)` | `(input, param)` |
| 返回值 | `np.ndarray` / `list` (测量结果) | `pyqpanda3.core.QProg` (量子程序) |
| 测量方式 | 在 circuit 函数内自行调用 `ProbsMeasure` | 由 Layer 内部自动处理测量 |
| Pauli 期望 | 不直接支持 | 通过 `pauli_str_dict` 参数支持 |
| QVM 类型 | 默认 CPUQVM | 支持 `"cpu"` 或 `"gpu"` |
| 初始化参数 | 无 initializer | 支持 `initializer` 参数 |
| `shots` | 在测量函数中指定 | 在 Layer 构造时指定 |
| CRX/CRY/CRZ 梯度 | 标准 parameter-shift | 使用公式 https://iopscience.iop.org/article/10.1088/1367-2630/ac2cb3 |

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

## VQCQCloudLayer

**VQC 模块云提交层** - 将 VQC Module 提交到本源量子云执行。前向和反向传播均在量子云上完成，本地不执行任何量子计算。

```python
pyvqnet.qnn.pq3.vqc_qcloud_layer.VQCQCloudLayer(
    vqc_module,
    qcloud_token,
    pauli_str_dict=None,
    shots=1000,
    name="",
    submit_kwargs={},
    query_kwargs={}
)
```

**参数**:
- `vqc_module` - VQC Module (来自 `pyvqnet.qnn.vqc`)，内部必须包含 `QMachine` 且 `save_ir=True`
- `qcloud_token` - 从 https://qcloud.originqc.com.cn/ 获取的 API Token
- `pauli_str_dict` - Pauli 算子字典，用于期望值计算，如 `{'Z0': 1, 'Z1': 1}`
- `shots` - 测量次数，默认: 1000
- `submit_kwargs` - 提交参数，默认: `{"test_qcloud_fake": True}`（测试模式）
- `query_kwargs` - 查询参数

**关键约束**:
1. VQC 模块中的 `QMachine` 必须设置 `save_ir=True`
2. VQC 模块中必须调用 `reset_states(batchsize)`
3. **不支持** VQC 模块中的 `MeasureAll` — 改用 `pauli_str_dict` 指定观测量
4. Token 通过 `os.getenv("QCLOUD_TOKEN")` 获取，**不要硬编码**
5. 梯度使用 parameter-shift 规则在量子云上计算

**VQCQCloudLayer vs QuantumBatchAsyncQcloudLayer**:

| 维度 | VQCQCloudLayer | QuantumBatchAsyncQcloudLayer |
|------|----------------|------------------------------|
| 电路定义 | VQC Module (高层 API) | 原始 QPanda QProg 函数 (底层 API) |
| 量子门 | 使用 VQC 门 (RX, RY, RZ, CNOT 等) | 使用 QPanda 原生门 |
| 训练参数 | VQC Module 的 Parameter | 通过 `para_num` 指定 |
| 输入编码 | 在 Module forward 中手动编码 | 在 circuit 函数中手动编码 |

**示例**:
```python
import os
import pyvqnet
from pyvqnet.qnn.vqc import QMachine, RX, U1, CNOT
from pyvqnet.qnn import Module
from pyvqnet.qnn.pq3 import VQCQCloudLayer

token = os.getenv("QCLOUD_TOKEN")  # 不要硬编码！

class QModel(Module):
    def __init__(self, num_wires, dtype):
        super(QModel, self).__init__()
        self.qm = QMachine(num_wires, dtype=dtype, save_ir=True)
        self.rx_layer = RX(has_params=True, trainable=False, wires=0)
        self.u1 = U1(has_params=True, trainable=True, wires=[1])
        self.cnot = CNOT(wires=[0, 1])

    def forward(self, x, *args, **kwargs):
        self.qm.reset_states(x.shape[0])
        self.rx_layer(params=x[:, [0]], q_machine=self.qm)
        self.cnot(q_machine=self.qm)
        self.u1(q_machine=self.qm)
        return x

qmodel = QModel(num_wires=2, dtype=pyvqnet.kcomplex64)
layer = VQCQCloudLayer(
    qmodel,
    token,
    pauli_str_dict={'Z0': 1, 'Z1': 1},
    shots=1000,
    submit_kwargs={"test_qcloud_fake": True},  # 测试模式
)
x = pyvqnet.tensor.QTensor([[0.5, 0.3], [0.5, 0.3]], requires_grad=True)
y = layer(x)
y.backward()
print(x.grad)
print(qmodel.u1.params.grad)
```

---

## VQCLayer（统一量子层）

> 2.18.2 新增。VQCLayer 将包含 QMachine 的 VQC Module 包装为可训练的量子层，通过 `submit_kwargs["backend"]` 选择执行后端。

```python
pyvqnet.qnn.pq3.vqclayer.VQCLayer(
    vqc_module,
    qcloud_token=None,
    shots=1000,
    diff_method="parameter_shift",
    submit_kwargs=None,
    query_kwargs=None,
    name="",
)
```

### 执行后端

通过 `submit_kwargs["backend"]` 选择：

| 后端 | 说明 |
|------|------|
| `vqnet_native`（默认） | VQNet 本地 autograd 梯度，不提交线路 |
| `qcloud_fake` | 提交到本地 pyqpanda3 QVM（FakeBackend），参数移位法梯度 |
| `qpanda_runtime` | 提交到 qpanda3_runtime 真实芯片 |
| `qcloud_service` | 提交到 QCloud 真实芯片 |

### 测量方式（互斥）

- `submit_kwargs["pauli_str_dict"]`：期望值测量
- `query_kwargs["measure_qubits"]`：概率测量

同时指定或都不指定会抛出 `ValueError`。

### 经典影子测量（Classical Shadow）

`submit_kwargs["shadow_config"]` 控制期望值测量方案（需配合 `pauli_str_dict`）：

| shadow_config | 模式 | shots 语义 | 说明 |
|---------------|------|-----------|------|
| 0（默认） | 完整测量 | 传统 MeasureAll | 精确期望值 |
| 1 | 随机经典影子 | shots = 总快照数 T | 每个快照电路运行 1 shot（PennyLane 惯例） |
| 2 | 去随机化经典影子 | shots = 调度长度 N | 贪心调度生成 N 个电路，每个 1 shot |

`submit_kwargs["shadow_seed"]`：固定影子配方种子（默认随机，层生命周期内固定 → 训练可复现）。

**约束**：
- `shadow_config != 0` 必须同时提供 `pauli_str_dict`
- `shadow_config != 0` 不支持 `qpanda_runtime` 后端（抛 ValueError）
- 支持后端：`vqnet_native`、`qcloud_fake`、`qcloud_service`
- 梯度路径不变：始终使用 mode-0 精确参数移位法（E[shadow] = exact）

### 噪声模型（pyqpanda3 0.4.0）

`submit_kwargs["noise_model"]`：传入 `QCloudNoiseModel` 实例，本地 CPUQVM 生成逼真噪声结果。

```python
from pyqpanda3.qcloud import QCloudNoiseModel

noise = QCloudNoiseModel(chip_id="WK_C180")
layer = VQCLayer(
    model,
    submit_kwargs={
        "backend": "qcloud_fake",
        "pauli_str_dict": {"Z0 Z1": 1},
        "noise_model": noise,
        "test_qcloud_fake": True,
    },
)
```

**约束**：
- `qcloud_fake` 后端：noise_model 传入本地 CPUQVM，生成带噪声的模拟结果
- `qcloud_service` 后端：noise_model 类型必须是 `QCloudNoiseModel`（`core.NoiseModel` 仅用于 qcloud_fake）；chip_id 必须是云模拟器（`full_amplitude`/`partial_amplitude`/`single_amplitude`），真机芯片 + 云噪声抛 ValueError

### 日志与配置透传（pyqpanda3 0.4.0）

| submit_kwargs 键 | 说明 |
|-------------------|------|
| `log_file` | 日志输出到文件（通过 `LogOutput.FILE`），与 `if_print_qcloud_log` 互斥，FILE 优先 |
| `point_label` | 透传到云后端提交 JSON，默认 1 |
| `enable_binary_encoding` | 启用二进制编码提交，默认 False |
| `is_optimization` | 是否线路优化，本地 CPUQVM 返回伪优化结果，默认 True |

### 完整 submit_kwargs 键表

| 键 | 类型 | 默认值 | 说明 |
|----|------|--------|------|
| `backend` | str | `"vqnet_native"` | 执行后端 |
| `pauli_str_dict` | dict | None | 期望值测量（与 measure_qubits 互斥） |
| `shadow_config` | int | 0 | 影子测量模式：0/1/2 |
| `shadow_seed` | int | 随机 | 影子配方种子 |
| `noise_model` | QCloudNoiseModel | None | 噪声模型 |
| `log_file` | str | `""` | 日志文件路径 |
| `point_label` | int | 1 | 云后端点标签 |
| `enable_binary_encoding` | bool | False | 二进制编码 |
| `is_amend` | bool | True | 纠错 |
| `is_mapping` | bool | True | 逻辑-物理映射 |
| `is_optimization` | bool | True | 线路优化 |
| `chip_id` | str | `"WK_C180"` | 芯片 ID |
| `set_specified_block` | list | [] | 指定芯片块 |
| `default_task_group_size` | int | 200 | 任务分组大小 |
| `test_qcloud_fake` | bool | False | 本地模拟器干跑（qcloud_fake 强制 True） |
| `fake_async_delay` | float | None | 模拟器异步延迟秒数 |
| `print_ir` | bool | False | 打印 OriginIR |
| `server_ip_address` | str | `""` | QCloud 服务器地址 |
| `use_qwc` | bool | True | QWC 任务分组 |
| `streaming` | bool | False | 流式梯度（提交-收缩重叠） |
| `block_b` / `block_o` / `block_k` | int | 0 | 块化收缩维度（0 不启用） |

### query_kwargs 键表

| 键 | 类型 | 默认值 | 说明 |
|----|------|--------|------|
| `measure_qubits` | list | None | 概率测量量子比特（与 pauli_str_dict 互斥） |
| `total_timeout` | int | 60 | 总超时秒数（本地模拟器 3600） |
| `timeout` | int | 1 | 单次查询超时（本地模拟器 0.01） |
| `print_query_info` | bool | False | 打印查询信息 |
| `sub_circuits_split_size` | int | 1 | 子线路拆分大小 |

### 示例：vqnet_native + 完整测量

```python
import pyvqnet
from pyvqnet import nn
from pyvqnet.tensor import QTensor
from pyvqnet.qnn.vqc import QMachine, RY
from pyvqnet.qnn.pq3 import VQCLayer

class QModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qm = QMachine(2, save_ir=True)
        self.ry_w = RY(trainable=True, wires=0, init_params=QTensor([[0.4]]))
        self.ry_x = RY(trainable=False, wires=1)

    def forward(self, x):
        self.qm.reset_states(x.shape[0])
        self.ry_w(q_machine=self.qm)
        self.ry_x(params=x[:, [0]], q_machine=self.qm)
        return x

layer = VQCLayer(
    QModel(),
    submit_kwargs={"backend": "vqnet_native", "pauli_str_dict": {"Z0 Z1": 1}},
)
x = QTensor([[0.3], [0.7]], requires_grad=True)
y = layer(x)
y.backward()
print(y)          # Z0*Z1 期望值, 形状 (2, 1)
print(x.grad)     # d<Z0 Z1>/dx
```

### 示例：经典影子测量

```python
import pyvqnet
from pyvqnet import nn
from pyvqnet.tensor import QTensor
from pyvqnet.qnn.vqc import QMachine, RY
from pyvqnet.qnn.pq3 import VQCLayer

class QModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qm = QMachine(2, save_ir=True)
        self.ry_w = RY(trainable=True, wires=0, init_params=QTensor([[0.4]]))
        self.ry_x = RY(trainable=False, wires=1)

    def forward(self, x):
        self.qm.reset_states(x.shape[0])
        self.ry_w(q_machine=self.qm)
        self.ry_x(params=x[:, [0]], q_machine=self.qm)
        return x

# 随机经典影子 (shadow_config=1)
layer = VQCLayer(
    QModel(), shots=500,
    submit_kwargs={
        "backend": "vqnet_native",
        "pauli_str_dict": {"Z0 Z1": 1},
        "shadow_config": 1,
        "shadow_seed": 42,  # 固定种子 → 训练可复现
    },
)
x = QTensor([[0.3], [0.7]], requires_grad=True)
y = layer(x)
y.backward()
print(x.grad)

# 去随机化影子 (shadow_config=2)
layer2 = VQCLayer(
    QModel(), shots=500,
    submit_kwargs={
        "backend": "vqnet_native",
        "pauli_str_dict": {"Z0 Z1": 1},
        "shadow_config": 2,
        "shadow_seed": 42,
    },
)
y2 = layer2(QTensor([[0.3], [0.7]], requires_grad=True))
```

### 示例：qcloud_fake + 噪声模型

```python
from pyqpanda3.qcloud import QCloudNoiseModel

noise = QCloudNoiseModel(chip_id="WK_C180")
layer = VQCLayer(
    QModel(),
    shots=8000,
    submit_kwargs={
        "backend": "qcloud_fake",
        "test_qcloud_fake": True,
        "pauli_str_dict": {"Z0 Z1": 1},
        "noise_model": noise,
    },
)
x = QTensor([[0.3]], requires_grad=True)
y = layer(x)
y.backward()
```

---

## TorchVQCLayer（torch 后端统一量子层）

> 2.18.2 新增。TorchVQCLayer 是 VQCLayer 的 torch 后端版本，功能与原生 VQCLayer 一致。输入/输出为 `torch.Tensor`，需先切换到 torch 后端：`pyvqnet.backends.set_backend("torch")`。

```python
pyvqnet.qnn.pq3.torch.torchvqclayer.TorchVQCLayer(
    vqc_module,
    qcloud_token=None,
    shots=1000,
    diff_method="parameter_shift",
    submit_kwargs=None,
    query_kwargs=None,
    name="",
)
```

**与 VQCLayer 的差异**：
- `vqc_module` 必须是 torch 后端 VQC Module（`pyvqnet.qnn.vqc.sv.torch` 构建）
- 输入/输出为 `torch.Tensor`（前向边界自动包装）
- `submit_kwargs` / `query_kwargs` 键与 VQCLayer 完全一致
- 经典影子（shadow_config）、噪声模型、日志/配置透传均支持
- 梯度路径：torch 后端 autograd（vqnet_native）或参数移位法（提交后端）

### 示例：torch 后端 + 经典影子

```python
import pyvqnet
pyvqnet.backends.set_backend("torch")

from pyvqnet import nn
from pyvqnet.tensor import QTensor
from pyvqnet.qnn.vqc.sv.torch import QMachine, RY
from pyvqnet.qnn.pq3.torch import TorchVQCLayer

class QModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qm = QMachine(2, save_ir=True)
        self.ry_w = RY(trainable=True, wires=0, init_params=QTensor([[0.4]]))
        self.ry_x = RY(trainable=False, wires=1)

    def forward(self, x):
        self.qm.reset_states(x.shape[0])
        self.ry_w(q_machine=self.qm)
        self.ry_x(params=x[:, [0]], q_machine=self.qm)
        return x

layer = TorchVQCLayer(
    QModel(), shots=500,
    submit_kwargs={
        "backend": "vqnet_native",
        "pauli_str_dict": {"Z0 Z1": 1},
        "shadow_config": 1,
        "shadow_seed": 42,
    },
)
x = QTensor([[0.3], [0.7]])
y = layer(x)             # y 是 torch.Tensor
print(y.detach().numpy())  # 用 .detach().numpy() 取值
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

## NoiseQuantumLayer

**噪声模拟量子层** - 在带噪声的量子虚拟机 (`pyqpanda.NoiseQVM`) 上模拟参数化量子电路，支持自定义噪声模型配置。

```python
pyvqnet.qnn.quantumlayer.NoiseQuantumLayer(
    qprog_with_measure,
    para_num,
    machine_type,
    num_of_qubits,
    num_of_cbits=1,
    diff_method="parameter_shift",
    delta=0.01,
    noise_set_config=None,
    dtype=None,
    name=""
)
```

**参数**:
- `qprog_with_measure` - 量子电路函数，**签名必须为 `(input, param, qubits, cbits, m_machine)`**
- `para_num` - 参数数量
- `machine_type` - 必须为 `"noise"`（目前仅支持噪声模拟）
- `num_of_qubits` - 量子比特数
- `num_of_cbits` - 经典比特数，默认: 1
- `diff_method` - 梯度方法: `"parameter_shift"` 或 `"finite_diff"`，默认: `"parameter_shift"`
- `delta` - 有限差分步长，默认: 0.01
- `noise_set_config` - 噪声配置函数，**签名: `def noise_set_config(qvm, qubits)`**，默认: 使用 BITFLIP_KRAUS_OPERATOR (p=0.01)
- `dtype` - 参数数据类型，默认: None
- `name` - 模块名称

**关键函数签名**:
```python
def qprog_with_measure(input, param, qubits, cbits, m_machine):
    # input: 一维经典输入数据
    # param: 一维变分参数
    # qubits: 由 NoiseQuantumLayer 分配的量子比特
    # cbits: 由 NoiseQuantumLayer 分配的经典比特
    # m_machine: 由 NoiseQuantumLayer 创建的噪声模拟器 (NoiseQVM)
    # 返回: 期望值 (float)
```

**内置默认噪声模型** (当 `noise_set_config=None` 时):
- BITFLIP_KRAUS_OPERATOR 应用于: X, Y, Z, RX, RY, RZ, H 门 (p=0.01)
- DAMPING_KRAUS_OPERATOR 应用于: CNOT 门 (p=0.01)

**示例**:
```python
from pyvqnet.qnn import NoiseQuantumLayer
from pyvqnet.tensor import QTensor
import pyqpanda as pq
import numpy as np

# 定义电路函数 (签名包含 qubits, cbits, m_machine)
def circuit(input, param, qubits, cbits, machine):
    cir = pq.QCircuit()
    cir.insert(pq.H(qubits[0]))
    cir.insert(pq.RY(qubits[0], input[0]))
    cir.insert(pq.RY(qubits[0], param[0]))
    prog = pq.QProg()
    prog.insert(cir)
    prog << pq.measure_all(qubits, cbits)

    result = machine.run_with_configuration(prog, cbits, 100)
    counts = np.array(list(result.values()))
    states = np.array(list(result.keys())).astype(float)
    probabilities = counts / 100
    expectation = np.sum(states * probabilities)
    return expectation

# 自定义噪声配置
def my_noise_config(qvm, qubits):
    p = 0.01
    from pyqpanda import NoiseModel, GateType
    qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.PAULI_X_GATE, p)
    qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.HADAMARD_GATE, p)
    qvm.set_noise_model(NoiseModel.DAMPING_KRAUS_OPERATOR, GateType.CNOT_GATE, p, [
        [qubits[i], qubits[i + 1]] for i in range(len(qubits) - 1)
    ])

qlayer = NoiseQuantumLayer(
    circuit, 24, "noise", 1, 1,
    diff_method="parameter_shift",
    delta=0.01,
    noise_set_config=my_noise_config,
)
input = QTensor([[0.0, 1.0, 1.0, 1.0]])
rlt = qlayer(input)
grad = QTensor(np.ones(rlt.data.shape) * 1000)
rlt.backward(grad)
print(qlayer.m_para.grad)
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
振幅编码。**输入特征必须手动归一化（L2 norm = 1）**，否则不是合法量子态。`n` 个量子比特编码 `2^n` 个特征。

```python
from pyvqnet.qnn.pq3.template import AmplitudeEmbeddingCircuit
import numpy as np
import pyqpanda3.core as pq

input_feat = np.array([2.2, 1, 4.5, 3.7])
# 必须手动归一化！L2 norm 必须为 1
input_feat = input_feat / np.linalg.norm(input_feat)
# 4 个特征 → 2 个量子比特 (2^2 = 4)
qlist = range(2)
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
from pyvqnet.tensor import tensor

hea = HardwareEfficientAnsatz(
    qubits=range(4),
    single_rot_gate_list=["RX", "RY", "RZ"],
    entangle_gate="CNOT",
    entangle_rules="linear",
    depth=2
)
# get_para_num() 获取所需参数总数
para_count = hea.get_para_num()
# create_ansatz(params) 需传入参数张量，形状 [para_count]
params = tensor.ones([para_count])
cir = hea.create_ansatz(params)
```

---

## 常见问题

1. **参数顺序错误**: 函数签名必须是 `(input, param)`，不是 `(param, input)`
2. **忘记返回测量结果**: 函数必须返回 `np.ndarray` 或 `list`
3. **QCloud Token**: 使用 `os.getenv("QCLOUD_TOKEN")`，不要硬编码
4. **梯度计算开销**: parameter-shift 需要额外运行 `para_num × batch_size × input_dim` 次电路
5. **AmplitudeEmbedding 未归一化**: 输入特征必须手动归一化 (`x / np.linalg.norm(x)`), L2 norm 必须为 1; 特征数必须 ≤ 2^n_qubits
6. **HardwareEfficientAnsatz 参数遗漏**: 必须调用 `get_para_num()` 获取参数总数, 再用 `create_ansatz(params)` 传入参数张量; 不能调用无参的 `create_ansatz()`
7. **VQCQCloudLayer QMachine 配置**: QMachine 必须设置 `save_ir=True`；VQC 模块不支持 MeasureAll，改用 `pauli_str_dict`
8. **NoiseQuantumLayer machine_type**: 目前仅支持 `"noise"` 类型；函数签名须包含 `(input, param, qubits, cbits, m_machine)` 五个参数
9. **QuantumLayerV3 CRX/CRY/CRZ 梯度**: 使用特殊公式计算，参考 https://iopscience.iop.org/article/10.1088/1367-2630/ac2cb3

---

**Version**: VQNet 2.18.1
