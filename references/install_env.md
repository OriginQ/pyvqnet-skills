# VQNet 安装与环境配置

> **重要**: 所有示例代码均来自官方文档。

---

## 安装要求

### Python 版本
- **Python 3.10～3.14**

### 支持平台
- Linux
- Windows
- macOS 13+ (arm64)

---

## 安装命令

```bash
pip install pyvqnet --upgrade
```

---

## CUDA GPU 支持

对于 Windows 和 Linux，pyvqnet 软件包自带基于 NVIDIA CUDA 的经典神经网络计算加速功能。

### CUDA 架构要求
- **sm_80**: NVIDIA A100, A30 系列数据中心 GPU
- **sm_86**: NVIDIA GeForce RTX 30 系列消费级 GPU

### CUDA 运行时依赖
软件包编译时针对 CUDA 12.6，自动安装以下依赖：

```
nvidia-cublas-cu12==12.6.4.1
nvidia-cuda-runtime-cu12==12.6.77
nvidia-nccl-cu12==2.28.9
nvidia-nvjitlink-cu12==12.6.85
nvidia-cuda-cupti-cu12==12.6.80
nvidia-cuda-nvrtc-cu12==12.6.85
nvidia-cufft-cu12==11.3.0.4
nvidia-cusolver-cu12==11.7.1.2
nvidia-cusparse-cu12==12.5.4.2
nvidia-nvtx-cu12==12.6.77
nvidia-curand-cu12==10.3.7.77
```

**注意**: 可能与依赖不同版本 CUDA 的其他软件产生冲突。

---

## 验证安装

### CPU 测试

```python
import pyvqnet
from pyvqnet.tensor import arange

a = arange(1, 25).reshape([2, 3, 4])
print(a)
# [
# [[1., 2., 3., 4.],
#  [5., 6., 7., 8.],
#  [9., 10., 11., 12.]],
# [[13., 14., 15., 16.],
#  [17., 18., 19., 20.],
#  [21., 22., 23., 24.]]
# ]
```

### GPU 测试

```python
from pyvqnet import DEV_GPU_0
from pyvqnet.tensor import ones

a = ones([4, 5], device=DEV_GPU_0)
print(a)
# [[1., 1., 1., 1., 1.],
#  [1., 1., 1., 1., 1.],
#  [1., 1., 1., 1., 1.],
#  [1., 1., 1., 1., 1.]]
```

---

## 常见安装问题

### 1. Windows DLL 加载失败

```
ImportError: DLL load failed while importing _core: 找不到指定的模块。
```

**解决方案**: 安装 VC++ 运行时库。
- 参考: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist

### 2. CUDA 版本冲突

如果与其他依赖 CUDA 12 的库（如 torch）冲突，建议：
- 使用 conda 管理环境，为 VQNet 创建独立环境
- 或使用 `pyvqnet.backends.set_backend("torch")` 切换后端

---

## 依赖安装

### pyqpanda3（量子计算后端）

VQNet 的量子计算模块依赖 pyqpanda3：

```bash
pip install pyqpanda3
```

### PyTorch Backend（可选）

自 v2.15.0 版本支持使用 PyTorch 作为计算后端，需自行安装 PyTorch（GPU 场景建议安装与 CUDA 12.x 匹配的版本；官方基准使用 torch 2.11.0+cu126 验证）：

```bash
pip install torch
```

使用方法：

```python
import pyvqnet.backends
pyvqnet.backends.set_backend("torch")
```

### 张量网络后端依赖（TN，可选）

张量网络（Tensor Network）后端基于 `jax` 实现自动微分与 GPU 加速，默认安装 `pyvqnet` 不包含该依赖：

```bash
# CPU
pip install jax
# GPU（需 CUDA 12.6）
pip install "jax[cuda12]"

# 张量网络库
pip install tensornetwork
```

### 新增依赖（v2.18.0+）

自 v2.18.0 版本，新增以下核心依赖：

| 依赖 | 用途 |
|------|------|
| `rustworkx>=0.13.2` | CircuitGraph 量子电路图数据结构 |
| `pydantic>=2.0.0` | ZeRO 分布式优化配置解析 |
| `packaging` | 版本号比较与依赖管理 |

---

## 数据类型说明

自 v2.0.7 版本，QTensor 增加了 dtype 属性，参照 PyTorch 对输入进行了限制：

| 用途 | dtype |
|------|-------|
| Embedding 层输入 | `kint64` |
| CrossEntropyLoss 标签 | `kint64` |
| CategoricalCrossEntropy 标签 | `kint64` |
| SoftmaxCrossEntropy 标签 | `kint64` |
| NLL_Loss 标签 | `kint64` |
| bfloat16 混合精度计算 | `kbfloat16`（v2.18.0+） |
| 普通张量 | `kfloat32`（默认） |

使用 `astype()` 进行类型转换：

```python
from pyvqnet.tensor import QTensor
from pyvqnet import kint64

labels = QTensor([0, 1, 2]).astype(kint64)
```

---

## 模型定义注意事项

### ModuleList vs Python List

在 `Module` 中使用多个子模块时，**必须使用 `ModuleList`**，不能使用 Python 的 `list`：

```python
from pyvqnet.nn import Module, Linear, ModuleList

class M(Module):
    def __init__(self):
        super().__init__()
        # 正确：使用 ModuleList
        self.layers = ModuleList([Linear(10, 20), Linear(20, 10)])
        # 错误：使用 list（参数不会被注册）
        # self.layers = [Linear(10, 20), Linear(20, 10)]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

---

## GPU 设备常量

```python
from pyvqnet import DEV_CPU, DEV_GPU_0, DEV_GPU_1
from pyvqnet.tensor import ones

# CPU
a = ones([4, 5], device=DEV_CPU)

# GPU
a = ones([4, 5], device=DEV_GPU_0)
```

---

## 本源量子云真机

VQNet 支持在本源量子云真机上运行量子电路：

1. 获取 API Token: https://qcloud.originqc.com.cn/
2. 使用 `QuantumBatchAsyncQcloudLayer`

```python
import os
from pyvqnet.qnn.pq3.quantumlayer import QuantumBatchAsyncQcloudLayer

token = os.getenv("QCLOUD_TOKEN")  # 不要硬编码！

layer = QuantumBatchAsyncQcloudLayer(
    circuit_func, token, param_num,
    submit_kwargs={"test_qcloud_fake": True}  # 测试模式
)
```

---

## 简单量子分类器示例

```python
import pyqpanda3.core as pq
from pyvqnet.qnn.pq3 import ProbsMeasure, QuantumLayer
from pyvqnet.nn import Module, CategoricalCrossEntropy
from pyvqnet.optim import Adam
from pyvqnet.tensor import QTensor
from pyvqnet import kint64
import numpy as np

def qdrl_circuit(input, weights):
    qlist = range(1)
    machine = pq.CPUQVM()
    x1 = input.squeeze()
    param1 = weights.squeeze()

    circult = pq.QCircuit()
    circult << pq.RZ(qlist[0], x1[0])
    circult << pq.RY(qlist[0], x1[1])
    circult << pq.RZ(qlist[0], x1[2])
    circult << pq.RZ(qlist[0], param1[0])
    circult << pq.RY(qlist[0], param1[1])
    circult << pq.RZ(qlist[0], param1[2])

    prog = pq.QProg()
    prog << circult
    prob = ProbsMeasure(machine, prog, qlist)
    return prob

class Model(Module):
    def __init__(self):
        super().__init__()
        self.pqc = QuantumLayer(qdrl_circuit, 3)  # 3个可训练参数: param1[0], param1[1], param1[2]

    def forward(self, x):
        return self.pqc(x)

model = Model()
optimizer = Adam(model.parameters(), lr=0.6)
loss_fn = CategoricalCrossEntropy()

# 训练循环
x = QTensor(np.random.randn(32, 3).astype(np.float32))  # numpy 默认 float64，需转 float32
y = QTensor(np.random.randint(0, 2, (32, 2)), dtype=kint64)  # 标签必须用 kint64，不要用 np.int64

output = model(x)
loss = loss_fn(y, output)  # 注意：(标签, 预测值)
optimizer.zero_grad()
loss.backward()
optimizer._step()
```

---

## FAQ

### Q: VQNet 有哪些特性？

VQNet 是基于本源量子 pyQPanda3 开发的量子机器学习工具集。提供丰富、易用的经典神经网络计算模块接口，可以方便地进行机器学习的优化，模型定义方式与主流机器学习框架一致，降低了用户学习成本。

### Q: 如何使用 VQNet 进行量子机器学习模型训练？

1. 通过 pyQPanda3 构建虚拟机，结合 VQNet 接口构建量子、量子经典混合模型 `Module`
2. 调用 `forward()` 进行量子线路模拟以及经典神经网络前向运算
3. 调用 `backward()` 进行自动微分，计算参数梯度
4. 结合优化器的 `_step()`（或等价的 `step()`）进行参数优化

### Q: 为什么定义的模型参数在训练时不更新？

可能原因：
- 模块不可微分（使用 VQNet 提供的接口）
- 子模块使用 Python `list` 而不是 `ModuleList`
- 未设置 `requires_grad=True`

### Q: VQNet 是否依赖 PyTorch？

VQNet 不依赖 PyTorch，也不自动安装 PyTorch。

自 v2.15.0 版本支持使用 PyTorch 作为计算后端，需自行安装 PyTorch。

---

## v2.18.2 新特性

VQNet v2.18.2 引入/调整以下重要特性：

### VQCLayer / TorchVQCLayer 统一量子层
- `pyvqnet.qnn.pq3.vqclayer.VQCLayer` 与 `pyvqnet.qnn.pq3.torch.TorchVQCLayer`，通过 `submit_kwargs["backend"]` 选择 `vqnet_native`（原 `vqc_autograd`）、`qcloud_fake`、`qcloud_service`、`qpanda_runtime` 四种执行后端。
- 支持期望值（`pauli_str_dict`）与概率（`measure_qubits`）两种互斥测量方式。

### 混合并行分布式训练
- 新增 `pyvqnet.distributed.hybrid.ParallelTrainingWrapper`，可将经典层与 VQC 混合编排为经典流水线并行（PP）、数据并行（DP）与比特并行。

### 分布式后端
- 移除 MPI 后端，CPU 集合通信统一使用 `gloo`；`CommController` 默认 backend 从 `mpi` 改为 `gloo`。
- GPU 通信使用 NCCL（随包附带）；`vqnetrun` 启动时需提供 `--nproc_per_node`。

### 量子接口同步
- 量子逻辑门类移除 `has_params` 参数，`trainable` 调整为第一位置参数；可训练门写作 `RX(trainable=True, wires=0)`。
- `QTensor` 移除 `name`、`nodes` 参数。
- `QuantumLayerAdjoint` 移除 `use_qpanda` 参数，`general_module` 重命名为 `vqc_module`。
- `TNQModule` 新增 `mesh`、`cotengra_options`、`tree_path` 参数，并修正 `use_jit` 默认值。
- `ConvT2D` 的 `padding` 不再支持 `"same"`，仅支持 `"valid"` 或整数元组。

### 文档与其他
- `QiskitLayer` / `CirqLayer` 接口文档同步；`SPSA`（`pyvqnet.qnn`）与 `QNSPSAOptimizer` 优化器文档同步，`SPSA._step()` 梯度更新问题修复。
- 文档中 `pyQPanda` 相关表述统一更新为 `pyQPanda3`。

### bfloat16 支持
新增 `kbfloat16` dtype（v2.18.0+），支持 bfloat16 混合精度训练，降低显存占用。

---

**Version**: VQNet 2.18.2
