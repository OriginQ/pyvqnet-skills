# VQNet 分布式计算模块 API Reference

> **重要**: 分布式计算功能**仅在 Linux 操作系统**下可用。

---

## 概述

分布式计算是指通过多台设备（如 GPU/CPU 节点）协同完成神经网络的训练或推理任务，利用并行处理加速计算并扩展模型规模。

VQNet 分布式计算模块：
- 使用 MPI 启动多进程并行计算
- 使用 NCCL 进行 GPU 之间通信
- 仅支持 Linux 系统

---

## 环境部署

### MPI 安装

VQNet CPU 分布式计算基于 MPI 实现。

**安装 mpicxx**:

```bash
conda install conda-forge::mpich-mpicxx==4.1.2
```

**安装 mpi4py**:

```bash
pip install mpi4py
```

**mpi4py 安装问题解决**:

如果出现兼容性问题，执行以下命令：

```bash
# 暂存编译器
pushd /root/anaconda3/envs/$CONDA_DEFAULT_ENV/compiler_compat && mv ld ld.bak && popd

# 再次安装
pip install mpi4py

# 还原
pushd /root/anaconda3/envs/$CONDA_DEFAULT_ENV/compiler_compat && mv ld.bak ld && popd
```

### NCCL 安装

VQNet GPU 分布式计算基于 NCCL 实现。软件包默认安装 NCCL 动态链接库。

**手动安装 NCCL**（可选）：

```bash
git clone https://github.com/NVIDIA/nccl.git
cd nccl
make -j src.build

# 如果 CUDA 未安装到默认路径
make src.build CUDA_HOME=<path to cuda install>
make src.build CUDA_HOME=<path to cuda install> BUILDDIR=/usr/local/nccl
```

**配置环境变量**:

```bash
vim ~/.bashrc

# 添加以下内容
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/lib
export PATH=$PATH:/usr/local/nccl/bin

source ~/.bashrc
```

### 节点间通信配置

多节点分布式计算需要：
- 多节点 mpich 环境一致
- Python 环境一致
- 节点间免密通信

**设置免密通信**:

```bash
# 在每个节点上执行
ssh-keygen

# 将其他节点公钥添加到主节点 authorized_keys
# 在子节点 node1 上执行
cat ~/.ssh/id_dsa.pub >> node0:~/.ssh/authorized_keys

# 在子节点 node2 上执行
cat ~/.ssh/id_dsa.pub >> node0:~/.ssh/authorized_keys

# 将 authorized_keys 拷贝到其他节点
scp ~/.ssh/authorized_keys node1:~/.ssh/authorized_keys
scp ~/.ssh/authorized_keys node2:~/.ssh/authorized_keys
```

---

## 启动命令 vqnetrun

使用 `vqnetrun` 命令启动分布式训练。

### 后端选择 (--backend)

选择分布式后端，支持 `mpi`（CPU）和 `nccl`（GPU），默认 `mpi`。

```bash
vqnetrun --backend mpi -np 4 python train.py    # MPI 模式（默认）
vqnetrun --backend nccl --nproc_per_node 2 python train.py  # NCCL 模式
```

### MPI 模式参数

#### 进程数 (-n/-np)

```bash
vqnetrun -n 2 python train.py
vqnetrun -np 4 python train.py
```

#### 节点指定 (-H/--hosts)

```bash
vqnetrun -np 4 -H node0:1,node2:1 python train.py
vqnetrun -np 4 --hosts node0:1,node2:1 python train.py
```

#### Hostfile (-f/-hostfile/--hostfile)

文件格式：
```
node0 slots=1
node2 slots=1
```

```bash
vqnetrun -np 4 -f hosts python train.py
vqnetrun -np 4 -hostfile hosts python train.py
vqnetrun -np 4 --hostfile hosts python train.py
```

### NCCL 模式参数

NCCL 模式使用去中心化启动模型（decentralized launch）：每个节点独立运行 `vqnetrun`，通过环境变量通信。**不支持** `-H/--hosts` 或 `--hostfile`。

#### 每节点进程数 (--nproc_per_node)

```bash
vqnetrun --backend nccl --nproc_per_node 4 python train.py
```

#### 节点总数 (--nnodes)

```bash
vqnetrun --backend nccl --nproc_per_node 4 --nnodes 2 --node_rank 0 python train.py
```

#### 节点编号 (--node_rank)

```bash
vqnetrun --backend nccl --nproc_per_node 4 --nnodes 2 --node_rank 1 python train.py
```

#### 主节点地址 (--master_addr) 和端口 (--master_port)

```bash
vqnetrun --backend nccl --nproc_per_node 4 --master_addr 192.168.1.100 --master_port 29500 python train.py
```

#### NCCL 网络接口 (--nccl_socket_ifname)

对应 `NCCL_SOCKET_IFNAME` 环境变量。

```bash
vqnetrun --backend nccl --nproc_per_node 4 --nccl_socket_ifname eth0 python train.py
```

### 通用参数

#### 输出保存 (--output-filename)

```bash
vqnetrun -np 4 --hostfile hosts --output-filename output python train.py
```

#### 详细输出 (--verbose)

```bash
vqnetrun -np 4 --hostfile hosts --verbose python train.py
```

#### 超时设置 (--start-timeout)

默认 30 秒。

```bash
vqnetrun -np 4 --start-timeout 10 python train.py
```

#### 构建检查 (--check-build / -cb)

显示已编译的框架和通信库状态（MPI/NCCL），无需指定进程数：

```bash
vqnetrun --check-build
vqnetrun -cb --verbose
```

#### 禁用缓存 (--disable-cache)

vqnetrun 默认每 60 分钟缓存一次初始化检查结果。设置此标志禁用缓存：

```bash
vqnetrun --disable-cache -np 4 python train.py
```

---

## CommController

```python
pyvqnet.distributed.ControlComm.CommController(backend, rank=None, world_size=None)
```

控制 CPU/GPU 数据通信的控制器。

**参数**:
- `backend` - "mpi" (CPU) 或 "nccl" (GPU)
- `rank` - 仅在非 pyvqnet 后端下使用
- `world_size` - 仅在非 pyvqnet 后端下使用

**示例**:
```python
from pyvqnet.distributed import CommController

# GPU 分布式
Comm_OP = CommController("nccl")

# CPU 分布式
Comm_OP = CommController("mpi")
```

### getRank()

获取当前进程号。

```python
from pyvqnet.distributed import CommController
Comm_OP = CommController("nccl")
rank = Comm_OP.getRank()
```

### getSize()

获取总进程数。

```python
size = Comm_OP.getSize()  # vqnetrun -n 2 -> 2
```

### getLocalRank()

获取当前机器上的进程号。

```python
local_rank = Comm_OP.getLocalRank()
```

### barrier()

同步操作。

```python
Comm_OP.barrier()
```

### get_device_num()

获取当前节点显卡数量（仅 GPU）。

```python
Comm_OP.get_device_num()
```

---

## 通信操作

### allreduce

```python
Comm_OP.allreduce(tensor, c_op="avg")
```

支持 "sum", "avg" 计算方式。

```python
from pyvqnet.distributed import CommController
from pyvqnet.tensor import tensor
import numpy as np

Comm_OP = CommController("mpi")
num = tensor.to_tensor(np.random.rand(1, 5))
print(f"rank {Comm_OP.getRank()} {num}")

Comm_OP.allreduce(num, "sum")
print(f"rank {Comm_OP.getRank()} {num}")
# vqnetrun -n 2 python test.py
```

### reduce

```python
Comm_OP.reduce(tensor, root=0, c_op="avg")
```

将数据归约到指定进程。

```python
Comm_OP.reduce(num, 1)
```

### broadcast

```python
Comm_OP.broadcast(tensor, root=0)
```

将指定进程数据广播到所有进程。

```python
Comm_OP.broadcast(num, 1)
```

### allgather

```python
Comm_OP.allgather(tensor)
```

将所有进程数据 gather 到一起。

```python
num = Comm_OP.allgather(num)
```

### send/recv (P2P)

```python
Comm_OP.send(tensor, dest)
Comm_OP.recv(tensor, source)
```

点对点通信。

```python
from pyvqnet.distributed import CommController, get_rank
from pyvqnet.tensor import tensor
import numpy as np

Comm_OP = CommController("mpi")
num = tensor.to_tensor(np.random.rand(1, 5))
recv = tensor.zeros_like(num)

if get_rank() == 0:
    Comm_OP.send(num, 1)
elif get_rank() == 1:
    Comm_OP.recv(recv, 0)
# vqnetrun -n 2 python test.py
```

---

## 进程组操作

### split_groups

```python
Comm_OP.split_groups(rankL)
```

划分多个通信组。

```python
from pyvqnet.distributed import CommController

Comm_OP = CommController("mpi")
groups = Comm_OP.split_groups([[0, 1], [2, 3]])
print(groups)
# [[<mpi4py.MPI.Intracomm object>, [0, 3]], [<mpi4py.MPI.Intracomm object>, [2, 1]]]
# mpirun -n 4 python test.py
```

### 组内通信

```python
Comm_OP.allreduce_group(tensor, c_op="avg", group=None)
Comm_OP.reduce_group(tensor, root=0, c_op="avg", group=None)
Comm_OP.broadcast_group(tensor, root=0, group=None)
Comm_OP.allgather_group(tensor, group=None)
```

```python
from pyvqnet.distributed import CommController, get_rank, get_local_rank
from pyvqnet.tensor import tensor
from pyvqnet import kcomplex64

Comm_OP = CommController("nccl")
groups = Comm_OP.split_groups([[0, 1]])

complex_data = tensor.QTensor([3+1j, 2, 1 + get_rank()], dtype=kcomplex64).reshape((3, 1)).toGPU(1000 + get_local_rank())

Comm_OP.allreduce_group(complex_data, c_op="sum", group=groups[0])
# vqnetrun -n 2 python test.py
```

---

## NCCL 异步操作

### nccl_async_all_gather

```python
Comm_OP.nccl_async_all_gather(output, input, group=None, async_op=False)
```

```python
from pyvqnet import tensor, pyvqnet, kcomplex64
from pyvqnet.distributed import CommController, get_rank, get_local_rank

Comm_OP = CommController("nccl")
complex_data = tensor.QTensor([3+1j, 2, 1 + get_rank()], dtype=kcomplex64).reshape((3, 1)).toGPU(1000 + get_local_rank())
out_data = tensor.empty([2, 3, 1], dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0 + get_local_rank())

work = Comm_OP.nccl_async_all_gather(out_data, complex_data, group=None, async_op=True)
work.wait()
```

### nccl_async_all_reduce

```python
Comm_OP.nccl_async_all_reduce(tensor, c_op="avg", group=None, async_op=False)
```

```python
from pyvqnet import tensor, pyvqnet
from pyvqnet.distributed import CommController, get_local_rank

Comm_OP = CommController("nccl")
complex_data = tensor.ones([500, 500], dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0 + get_local_rank())

work = Comm_OP.nccl_async_all_reduce(complex_data, "sum", group=None, async_op=True)
work.wait()
```

### nccl_async_reduce

```python
Comm_OP.nccl_async_reduce(tensor, dest, c_op="avg", group=None, async_op=False)
```

### nccl_async_broadcast

```python
Comm_OP.nccl_async_broadcast(tensor, src, group=None, async_op=False)
```

### nccl_async_send/recv

```python
Comm_OP.nccl_async_send(tensor, dest, async_op=False)
Comm_OP.nccl_async_recv(tensor, source, async_op=False)
```

---

## 完整分布式训练示例

```python
from pyvqnet.distributed import CommController, get_rank, get_local_rank
from pyvqnet.nn import Module, Linear
from pyvqnet.optim import Adam
from pyvqnet.tensor import tensor, QTensor
from pyvqnet import DEV_GPU_0
import numpy as np

# 初始化通信控制器
Comm_OP = CommController("nccl")
rank = get_rank()
local_rank = get_local_rank()

# 设置 GPU
device = DEV_GPU_0 + local_rank

class Model(Module):
    def __init__(self):
        super().__init__()
        self.fc = Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

model = Model().toGPU(device)
optimizer = Adam(model.parameters(), lr=0.01)

# 数据加载（每个进程处理不同数据）
batch_size = 32
x = tensor.randn([batch_size, 10], device=device)
x.requires_grad = True
y = tensor.randn([batch_size, 5], device=device)

# 前向传播
pred = model(x)
loss = ((pred - y) ** 2).mean()

# 反向传播
optimizer.zero_grad()
loss.backward()

# 梯度同步
for param in model.parameters():
    Comm_OP.allreduce(param.grad, "avg")

optimizer._step()

if rank == 0:
    print(f"Loss: {loss.item()}")
```

---

## 辅助函数

```python
from pyvqnet.distributed import get_rank, get_local_rank, get_host_name, init_group

get_rank()        # 获取当前进程号
get_local_rank()  # 获取当前机器进程号
get_host_name()   # 获取主机名
init_group(rankL) # 初始化进程组
```

---

## Pipeline Parallel（流水线并行）

Pipeline Parallel（流水线并行）将模型按层切分为多个 stage（阶段），每个 stage 分配到不同的 GPU 上执行，实现模型并行训练。采用 1F1B（One-Forward-One-Backward）调度策略，有效降低流水线气泡。

**何时使用**：模型过大无法放入单个 GPU 显存，或需要将模型的不同层分布在多 GPU 上执行。

### PipelineModule

将模型定义为一个层序列，自动分配到各 pipeline stage：

```python
from pyvqnet.distributed.pp.pp_module import PipelineModule, LayerSpec, TiedLayerSpec

PipelineModule(layers, num_stages=None, topology=None, loss_fn=None,
               base_seed=1234, partition_method="parameters")
```

**参数**：
- `layers` - 层序列（`nn.Sequential` 或层列表）
- `num_stage` - pipeline stage 数量
- `topology` - `PipeDataParallelTopology`，指定 pipeline 和数据并行拓扑
- `loss_fn` - 损失函数
- `partition_method` - 层划分方法：`"uniform"`（均匀）、`"parameters"`（按参数量）、`"type:<name>"`（按类型）

### LayerSpec / TiedLayerSpec

用于延迟构建的层规格说明：

```python
LayerSpec(typename, *module_args, **module_kwargs)
TiedLayerSpec(key, typename, *module_args, forward_fn=None,
              tied_weight_attr=["weight"], **module_kwargs)
```

### PipelineParallelTrainingWrapper

高级训练封装，自动管理 engine/optimizer/dataloader：

```python
from pyvqnet.distributed.pp import PipelineParallelTrainingWrapper

PipelineParallelTrainingWrapper(args, join_layers, trainset)
```

**参数说明 (args dict)**：
- `"backend"` - `"nccl"`（仅 GPU 支持）
- `"train_batch_size"` - 全局 batch size
- `"train_micro_batch_size_per_gpu"` - 每 GPU micro batch size
- `"epochs"` - 训练轮数
- `"optimizer"` - 优化器配置 `{"type": "Adam", "params": {"lr": 0.001}}`
- `"pipeline_parallel_size"` - pipeline stage 数量
- `"loss"` - 损失函数实例
- `"steps"` - 每 epoch 步数
- `"partition_method"` - （可选）层划分方法

**示例**：

```python
import pyvqnet.nn as nn
from pyvqnet.nn import Sequential, CrossEntropyLoss, Linear, Conv2D
from pyvqnet.nn import activation as F, MaxPool2D
from pyvqnet.distributed.pp import PipelineParallelTrainingWrapper
from pyvqnet.distributed.configs import comm as dist
from pyvqnet.tensor import tensor

pipeline_parallel_size = 2
num_steps = 1000

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Sequential(
            Conv2D(3, 8, 3, padding='same'), F.ReLu(), MaxPool2D([2, 2], [2, 2]),
            Conv2D(8, 16, 3, padding='same'), F.ReLu(), MaxPool2D([2, 2], [2, 2]),
        )
        self.cls = Sequential(
            Linear(16 * 54 * 54, 64), F.ReLu(), Linear(64, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = tensor.flatten(x, 1)
        x = self.cls(x)
        return x

def join_layers(model):
    return [*model.features, lambda x: tensor.flatten(x, 1), *model.cls]

args = {
    "backend": "nccl",
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 32,
    "epochs": 5,
    "optimizer": {"type": "Adam", "params": {"lr": 0.001}},
    "pipeline_parallel_size": pipeline_parallel_size,
    "loss": CrossEntropyLoss(),
    "steps": num_steps,
}

trainset = ...  # 数据集
wrapper = PipelineParallelTrainingWrapper(args, join_layers(Model()), trainset)
wrapper.train_batch()
```

**注意事项**：
- 仅支持 Linux + GPU（NCCL）环境
- 模型需表示为层序列，`join_layers` 函数将模型展平为层列表
- 每个 stage 自动分配到对应 GPU

---

## 关于"混合并行"

VQNet 2.18.1 **未提供** `pyvqnet.distributed.hybrid` 混合并行模块。

如需数据并行 + 流水线并行组合的大规模训练，请使用流水线并行封装 `pyvqnet.distributed.pp.PipelineParallelTrainingWrapper`（见上文「Pipeline Parallel（流水线并行）」章节）。

---

## Tensor Parallel（张量并行）

Tensor Parallel 将线性层的权重矩阵沿行或列方向切分到多个 GPU，实现模型并行训练。

**何时使用**：单层（如 Linear）的参数量过大，无法放入单个 GPU，需要将矩阵运算拆分到多个设备。

### 导入路径

```python
# 直接从 distributed 包导入
from pyvqnet.distributed import ColumnParallelLinear, RowParallelLinear

# 或从 submodule 导入
from pyvqnet.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
```

### ColumnParallelLinear

沿列方向并行化的线性层：`Y = XA + b`，其中 A 按列切分为 `[A_1, ..., A_p]`。

```python
ColumnParallelLinear(input_size, output_size, weight_initializer=None,
                     bias_initializer=None, use_bias=True, dtype=None,
                     name="", tp_comm=None)
```

- `tp_comm` - 必须传入 `CommController` 实例用于通信

### RowParallelLinear

沿行方向并行化的线性层：`Y = XA + b`，其中 A 按行切分。

```python
RowParallelLinear(input_size, output_size, weight_initializer=None,
                  bias_initializer=None, use_bias=True, dtype=None,
                  name="", tp_comm=None)
```

**示例**：

```python
import pyvqnet
from pyvqnet.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from pyvqnet.distributed import CommController
from pyvqnet.device import DEV_GPU

pyvqnet.utils.set_random_seed(42)

Comm_OP = CommController("nccl")

fc1 = ColumnParallelLinear(8, 6, tp_comm=Comm_OP)
fc2 = RowParallelLinear(6, 4, tp_comm=Comm_OP)

z = pyvqnet.tensor.ones([2, 8], device=Comm_OP.get_rank() + DEV_GPU)
z.requires_grad = True

fc1 = fc1.to(Comm_OP.get_rank() + DEV_GPU)
fc2 = fc2.to(Comm_OP.get_rank() + DEV_GPU)

y = fc2(fc1(z))
y.backward()
```

---

## ZeRO Optimization（ZeRO 优化器）

ZeRO（Zero Redundancy Optimizer）通过切分优化器状态、梯度或参数来大幅降低分布式训练的内存占用。

### 支持阶段

| 阶段 | ZeroStageEnum | 说明 |
|------|---------------|------|
| Stage 0 | `disabled` | 无优化，等同标准分布式训练 |
| Stage 1 | `optimizer_states` | 切分优化器状态（Adam 动量等） |
| Stage 2 | `gradients` | 切分优化器状态 + 梯度 |
| Stage 3 | `weights` | 切分优化器状态 + 梯度 + 参数 |

### 导入路径

```python
from pyvqnet.distributed import (
    ZeroModelInitial,
    DeepSpeedZeroOptimizer,
    DeepSpeedZeroConfig,
    ZeroStageEnum,
    DummyOptim,
)
```

### DeepSpeedZeroConfig

ZeRO 配置模型，控制各阶段行为：

```python
from pyvqnet.distributed import DeepSpeedZeroConfig, ZeroStageEnum

config = DeepSpeedZeroConfig(
    stage=ZeroStageEnum.optimizer_states,  # Stage 1
    contiguous_gradients=True,             # 连续梯度缓冲区，减少碎片
    reduce_scatter=True,                   # 使用 reduce scatter
    reduce_bucket_size=500000000,          # 归约 bucket 大小（元素数）
    allgather_bucket_size=5000000000,      # allgather bucket 大小（元素数）
    overlap_comm=True,                     # 梯度通信与计算重叠
)
```

### ZeroModelInitial

ZeRO Stage 1 & 2 模型封装入口：

```python
ZeroModelInitial(args=None, model=None, optimizer=None)
```

**参数说明 (args dict)**：
- `"train_batch_size"` - 训练 batch size
- `"optimizer"` - 优化器配置 `{"type": "adam", "params": {"lr": 0.001}}`
- `"zero_optimization"` - ZeRO 配置 `{"stage": 1}` 或 `{"stage": 2}`

**使用流程**：`model.backward(loss)` → `model.step()`

**示例 (Stage 1)**：

```python
import pyvqnet
import pyvqnet.nn as nn
import pyvqnet.optim as optim
from pyvqnet import kint64, kfloat32
from pyvqnet.tensor import QTensor
from pyvqnet.distributed import (
    CommController, ZeroModelInitial, get_rank, get_local_rank
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)
        self.ac = nn.activation.ReLu()

    def forward(self, x):
        x = x.reshape([-1, 784])
        x = self.ac(self.fc1(x))
        x = self.ac(self.fc2(x))
        x = self.ac(self.fc3(x))
        x = self.ac(self.fc4(x))
        x = self.fc5(x)
        return x

local_rank = get_local_rank()
model = Model().to(local_rank + 1000)

Comm_OP = CommController("nccl")
Comm_OP.broadcast_model_params(model, 0)

batch_size = 64
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

args = {
    "train_batch_size": batch_size,
    "optimizer": {"type": "adam", "params": {"lr": 0.001}},
    "zero_optimization": {"stage": 1},  # 启用 ZeRO Stage 1
}

model = ZeroModelInitial(args=args, model=model, optimizer=optimizer)

# 训练循环
for epoch in range(5):
    model.train()
    for i in range(num_batches):
        data = QTensor(train_data[i], dtype=kfloat32).to(local_rank + 1000)
        labels = QTensor(train_labels[i], dtype=kint64).to(local_rank + 1000)

        outputs = model(data)
        loss = criterion(labels, outputs)

        model.backward(loss)  # 使用 model.backward 而非 loss.backward
        model.step()          # 使用 model.step 而非 optimizer._step()
```

---

## DataParallelVQCLayer（VQC 数据并行层）

提供 VQC（变分量子电路）的数据并行封装，自动处理梯度同步。支持普通模式和 adjoint 模式。

### DataParallelVQCLayer

标准 VQC 数据并行层：

```python
from pyvqnet.distributed import DataParallelVQCLayer

DataParallelVQCLayer(Comm_OP, vqc_module, name="",
                     use_bucket_cpp_allreduce=False,
                     bucket_bytes_cap=25 * 1024 * 1024,
                     first_bucket_bytes_cap=1024 * 1024,
                     find_unused_parameters=False,
                     gradient_as_bucket_view=False)
```

- `Comm_OP` - `CommController` 实例
- `vqc_module` - 包含 `forward()` 的 VQC Module（需正确设置 QMachine）

### DataParallelVQCAdjointLayer

使用 adjoint 梯度的 VQC 数据并行层（继承自 `QuantumLayerAdjoint`）：

```python
from pyvqnet.distributed import DataParallelVQCAdjointLayer

DataParallelVQCAdjointLayer(Comm_OP, vqc_module, name="",
                            use_bucket_cpp_allreduce=False,
                            bucket_bytes_cap=25 * 1024 * 1024,
                            first_bucket_bytes_cap=1024 * 1024,
                            find_unused_parameters=False,
                            gradient_as_bucket_view=False)
```

### DataParallelLayer

通用 VQC 数据并行层（与 DataParallelVQCLayer API 相同）：

```python
from pyvqnet.distributed import DataParallelLayer
```

### C++ Bucketing Allreduce

当 `use_bucket_cpp_allreduce=True` 时，需在 forward/backward 前后调用：

```python
layer = DataParallelVQCLayer(Comm_OP, vqc_module, use_bucket_cpp_allreduce=True)
y = layer(x)                        # forward
layer.prepare_for_backward(y)       # backward 前准备
y.backward()
layer.finalize_backward()           # backward 后结束
```

**示例**：

```python
import pyvqnet
from pyvqnet.qnn.vqc import QMachine, cnot, rx, rz, ry, MeasureAll, PauliX
from pyvqnet.tensor import tensor
from pyvqnet.nn import Module
from pyvqnet.distributed import CommController, DataParallelVQCLayer

class QModel(Module):
    def __init__(self, num_wires, num_layer, dtype, grad_mode=""):
        super().__init__()
        self.qm = QMachine(num_wires, dtype=dtype, grad_mode=grad_mode)
        self.measure = MeasureAll(obs=PauliX)
        self.n, self.l = num_wires, num_layer

    def forward(self, param, *args, **kwargs):
        self.qm.reset_states(param.shape[0])
        for j in range(self.l):
            cnot(self.qm, wires=[j, (j + 1) % self.l])
            for i in range(self.n):
                rx(self.qm, i, param[:, 3 * self.n * j + i])
                rz(self.qm, i, param[:, 3 * self.n * j + i + self.n])
                rx(self.qm, i, param[:, 3 * self.n * j + i + 2 * self.n])
        return self.measure(self.qm)

n, b, l = 4, 4, 2
input = tensor.ones([b, 3 * n * l])
input.requires_grad = True

Comm = CommController("mpi")
qmodel = QModel(num_wires=n, num_layer=l, dtype=pyvqnet.kcomplex64)

layer = DataParallelVQCLayer(Comm, qmodel)
y = layer(input)
y.backward()
```

---

## Qubits Reorder（量子比特重排）

Qubits Reorder 模块提供跨设备的量子比特分布执行，支持将大规模量子电路拆分到多个计算节点上运行。

### 导入路径

```python
# 核心分布式量子虚拟机
from pyvqnet.distributed.qubits_reorder import (
    DistributedQMachine,          # 分布式量子虚拟机
    QubitReorderLocalQmachine,    # 本地量子机（重排后）
    DistQuantumLayerAdjoint,      # 分布式量子层（adjoint 梯度）
    TorchDistributedQMachine,     # PyTorch 后端分布式量子虚拟机
    TorchQubitReorderLocalQmachine,
    TorchDistQuantumLayerAdjoint,
)

# 量子比特重排操作
from pyvqnet.distributed.qubits_reorder import (
    QubitReorder,                 # 量子比特重排器
    QubitReorderOp,               # 重排操作
    QubitsPermutation,            # 比特置换描述
    compute_qubit_mapping,        # 计算比特映射
    compute_qubit_remapping,      # 计算重映射
    flat_time_space_tiling,       # 时空平铺映射
    gen_qr_machine,               # 生成重排机器
)

# 量子比特间 swap 操作
from pyvqnet.distributed.qubits_reorder import (
    swap_global_global_q1,        # 全局-全局 swap
    swap_local_global_q1,         # 本地-全局 swap
)

# 工具函数
from pyvqnet.distributed.qubits_reorder import (
    all_same_qubit_dist_expval,   # 分布式期望值计算
    gen_local_qm_and_op_history,  # 生成本地量子机和操作历史
    measure_all_for_qr,           # 重排后测量
    exec_states_qr,               # 重排后执行态
    update_global_qmachine_states_in_qr,
    update_local_states_with_qr,
)
```

### 核心理念

Qubits Reorder 通过以下机制实现跨设备量子电路执行：
1. **量子比特映射**：将逻辑量子比特映射到物理设备
2. **时空平铺**：`flat_time_space_tiling` 将量子比特在时间-空间维度上展开
3. **SWAP 网络**：通过 `swap_global_global_q1` / `swap_local_global_q1` 在设备间交换量子比特
4. **分布式期望值**：`all_same_qubit_dist_expval` 跨设备计算期望值

### DistQuantumLayerAdjoint

将 QuantumLayer 包装为分布式执行版本（基于量子比特重排）：

```python
from pyvqnet.distributed.qubits_reorder import DistQuantumLayerAdjoint

adjoint_layer = DistQuantumLayerAdjoint(quantum_layer_module)
```

### DistributedQMachine

分布式量子虚拟机，管理跨设备的量子态：

```python
from pyvqnet.distributed.qubits_reorder import DistributedQMachine

# 通过 set_qr_config 配置量子比特重排
```

---

## 高级函数参考

### gradient_allreduce

```python
from pyvqnet.distributed import all_grad_all_reduce, post_grad_all_reduce

all_grad_all_reduce(model, Comm_OP)        # 梯度全局 allreduce
post_grad_all_reduce(model, Comm_OP)       # 梯度后处理
```

### broadcast_model_params

```python
Comm_OP.broadcast_model_params(model, src=0)  # 广播模型参数
```

### 环境变量

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `LOCAL_RANK` | 本地进程编号 | 自动 |

---

## 常见问题

### Q: 分布式功能支持哪些系统？

**仅支持 Linux**。

### Q: 如何选择 MPI vs NCCL？

- **MPI**: CPU 分布式计算
- **NCCL**: GPU 分布式计算

### Q: 多节点运行时需要注意什么？

- 所有节点 Python 环境、mpich 版本一致
- 设置节点间免密通信
- 使用共享目录（如 NFS）确保文件同步

---

**Version**: VQNet 2.0
