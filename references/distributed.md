# VQNet 分布式计算模块 API Reference

> **重要**: 分布式计算功能**仅在 Linux 操作系统**下可用。
>
> **版本**: VQNet 2.18.2。本版本已全面改为 **gloo（CPU）/ NCCL（GPU）** 通信，使用 **spawn 子进程**方式启动多进程并行计算。

---

## 概述

分布式计算是指通过多台设备（如 GPU/CPU 节点）协同完成神经网络的训练或推理任务，利用并行处理加速计算并扩展模型规模。其核心是通过分布式接口（如 GLOO、NCCL）协调设备间的通信与同步。

VQNet 分布式计算模块的关键特性：

| 项目 | 说明 |
|------|------|
| 进程启动方式 | spawn 子进程（由 `vqnetrun` 启动器完成） |
| CPU 集合通信 | **gloo** |
| GPU 集合通信 | **NCCL** |
| 后端选择 | 在代码中由 `CommController` 决定，启动器不提供后端开关 |
| 支持平台 | **仅 Linux** |

---

## 环境部署

### NCCL 安装

NCCL 为 GPU 间通信的常用库，**VQNet 中 GPU 的分布式计算功能基于 NCCL 实现**。本软件默认在安装时同时安装 NCCL 的动态链接库，一般不需要额外安装 NCCL。

如需手动构建 NCCL（可选）：

```bash
git clone https://github.com/NVIDIA/nccl.git
cd nccl
make -j src.build

# 如果 CUDA 未安装到默认路径
make src.build CUDA_HOME=<path to cuda install>
make src.build CUDA_HOME=<path to cuda install> BUILDDIR=/usr/local/nccl
```

配置环境变量：

```bash
vim ~/.bashrc

# 添加以下内容
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/lib
export PATH=$PATH:/usr/local/nccl/bin

source ~/.bashrc
```

> CPU 端使用内置的 gloo，无需额外安装通信库。

---

## 启动命令 vqnetrun

使用分布式计算接口时，通过 `vqnetrun` 命令启动。注意：`vqnetrun` 只负责启动多进程，**通信后端在代码中指定**。

### --nproc_per_node（必填）

`vqnetrun` **必须**通过 `--nproc_per_node` 参数指定当前节点启动的 worker 进程数。该参数同时适用于 gloo CPU worker 和 NCCL GPU worker。

```python
from pyvqnet.distributed import CommController

Comm_OP = CommController("gloo")  # 初始化 gloo 控制器

rank = Comm_OP.getRank()
size = Comm_OP.getSize()
print(f"rank: {rank}, size {size}")

# vqnetrun --nproc_per_node 2 python test.py
```

### 后端选择（在代码中完成）

分布式后端的选取在代码中完成，由 `CommController` 的构造参数决定：

- `CommController("gloo")` —— CPU 通信
- `CommController("nccl")` —— GPU 通信
- `CommController()` —— 根据要通信的数据所在设备自动决定通信方式

`vqnetrun` 命令**不提供**后端命令行参数。多进程启动时，每个 worker 进程自行确定后端：存在 CUDA 上下文时使用 NCCL 通信域，纯 CPU 运行时使用 gloo。

```python
from pyvqnet.distributed import CommController

Comm_OP = CommController("nccl")

rank = Comm_OP.getRank()
size = Comm_OP.getSize()
print(f"rank: {rank}, size {size}")

# 单机多 GPU:
# vqnetrun --nproc_per_node 2 python test.py

# 多节点情况下, 每个节点独立执行:
# 节点0: vqnetrun --nproc_per_node 2 --nnodes 2 --node_rank 0 --master_addr 10.0.0.1 python test.py
# 节点1: vqnetrun --nproc_per_node 2 --nnodes 2 --node_rank 1 --master_addr 10.0.0.1 python test.py
```

### --nnodes

`--nnodes` 控制总节点数，默认值为 1。启动器使用 `nnodes * nproc_per_node` 计算 `WORLD_SIZE`。

```python
from pyvqnet.distributed import CommController

Comm_OP = CommController("nccl")
print(f"rank: {Comm_OP.getRank()}, size {Comm_OP.getSize()}")

# vqnetrun --nproc_per_node 2 --nnodes 2 --node_rank 0 --master_addr 10.0.0.1 python test.py
```

### --node_rank

`--node_rank` 控制当前节点的 rank，默认值为 0。启动器将其与 `nproc_per_node` 一起用于计算每个 worker 的全局 `RANK`。

```python
from pyvqnet.distributed import CommController

Comm_OP = CommController("nccl")
print(f"rank: {Comm_OP.getRank()}, size {Comm_OP.getSize()}")

# 节点0: vqnetrun --nproc_per_node 2 --nnodes 3 --node_rank 0 --master_addr 10.0.0.1 python test.py
# 节点1: vqnetrun --nproc_per_node 2 --nnodes 3 --node_rank 1 --master_addr 10.0.0.1 python test.py
# 节点2: vqnetrun --nproc_per_node 2 --nnodes 3 --node_rank 2 --master_addr 10.0.0.1 python test.py
```

### --master_addr

`--master_addr` 指定主节点地址，默认值为 `127.0.0.1`。该值会以 `MASTER_ADDR` 环境变量的形式传递给 worker 进程。

```python
from pyvqnet.distributed import CommController

Comm_OP = CommController("nccl")
print(f"rank: {Comm_OP.getRank()}, size {Comm_OP.getSize()}")

# vqnetrun --nproc_per_node 2 --nnodes 2 --node_rank 0 --master_addr 10.0.0.1 python test.py
```

### --master_port

`--master_port` 指定主节点端口，默认值为 `29500`。该值会以 `MASTER_PORT` 环境变量的形式传递给 worker 进程。

```python
from pyvqnet.distributed import CommController

Comm_OP = CommController("nccl")
print(f"rank: {Comm_OP.getRank()}, size {Comm_OP.getSize()}")

# vqnetrun --nproc_per_node 2 --master_port 29600 python test.py
```

### --nccl_socket_ifname

`--nccl_socket_ifname` 指定 NCCL 使用的网络接口名。该参数仅作用于 NCCL 通信，对应环境变量 `NCCL_SOCKET_IFNAME`，用于在多网卡环境下指定 NCCL 通信使用的网卡。

```python
from pyvqnet.distributed import CommController

Comm_OP = CommController("nccl")
print(f"rank: {Comm_OP.getRank()}, size {Comm_OP.getSize()}")

# vqnetrun --nproc_per_node 2 --nccl_socket_ifname eth0 python test.py
```

### --verbose

`--verbose` 输出额外的启动诊断信息，例如节点/进程 rank 设置、master 会合地址与端口、本地 worker 启动信息等。该标志**不执行**节点连通性检测。

```python
from pyvqnet.distributed import CommController

Comm_OP = CommController("gloo")
print(f"rank: {Comm_OP.getRank()}, size {Comm_OP.getSize()}")

# vqnetrun --nproc_per_node 4 --verbose python test.py
```

### -cb / --check-build

`-cb` 或 `--check-build` 查看当前构建支持的分布式控制器与通信库。由于启动器的参数解析要求必须提供 `--nproc_per_node`，查看构建信息时也需要传入该参数（取值为 1）。

```bash
vqnetrun --nproc_per_node 1 -cb
vqnetrun --nproc_per_node 1 --check-build
```

### 参数速查表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--nproc_per_node` | **必填** | 当前节点启动的 worker 进程数（gloo/NCCL 均需要） |
| `--nnodes` | 1 | 总节点数，`WORLD_SIZE = nnodes * nproc_per_node` |
| `--node_rank` | 0 | 当前节点 rank |
| `--master_addr` | 127.0.0.1 | 主节点地址，导出为 `MASTER_ADDR` |
| `--master_port` | 29500 | 主节点端口，导出为 `MASTER_PORT` |
| `--nccl_socket_ifname` | 无 | NCCL 网卡名，导出为 `NCCL_SOCKET_IFNAME` |
| `--verbose` | 关闭 | 输出额外启动诊断信息 |
| `-cb` / `--check-build` | 关闭 | 查看构建信息，需配合 `--nproc_per_node 1` |

---

## CommController

```python
pyvqnet.distributed.ControlComm.CommController(backend=None, rank=None, world_size=None)
```

控制 CPU/GPU 数据通信的控制器，通过 `backend` 生成 CPU（gloo）或 GPU（NCCL）控制器。

**参数**：

| 参数 | 说明 |
|------|------|
| `backend` | `"gloo"`（CPU）、`"nccl"`（GPU）；为 `None` 时根据通信数据所在设备自动选择，默认 `None` |
| `rank` | 当前进程号，仅在非 pyvqnet 后端（如 torch 后端）下使用，默认 `None` |
| `world_size` | 全局进程数，仅在非 pyvqnet 后端（如 torch 后端）下使用，默认 `None` |

**示例**：

```python
from pyvqnet.distributed import CommController

# GPU 分布式
Comm_OP = CommController("nccl")

# CPU 分布式
Comm_OP = CommController("gloo")

# 自动选择: 根据数据所在设备决定 gloo / nccl
Comm_OP = CommController()
```

### getRank()

获取当前进程号。

```python
from pyvqnet.distributed import CommController

Comm_OP = CommController("nccl")
rank = Comm_OP.getRank()
# vqnetrun --nproc_per_node 2 python test.py
```

### getSize()

获取总进程数。

```python
size = Comm_OP.getSize()   # vqnetrun --nproc_per_node 2 -> 2
```

### getLocalRank()

获取当前机器上的当前进程号。

```python
local_rank = Comm_OP.getLocalRank()
# vqnetrun --nproc_per_node 2 python test.py
```

### barrier()

同步操作。

```python
Comm_OP.barrier()
```

### get_device_num()

获取当前节点上的显卡数量（仅 GPU 下使用）。

```python
Comm_OP.get_device_num()
# vqnetrun --master_port 12345 --nproc_per_node 2 python test.py
```

### destroy()

销毁 NCCL 通信器资源。在进程退出前应调用此方法以正确清理 NCCL 资源；若不调用，NCCL 通信器可能泄漏内存，或在 CUDA 驱动关闭时引发错误。

```python
from pyvqnet.distributed import CommController

Comm_OP = CommController("nccl")
# ... 使用 Comm_OP ...
Comm_OP.destroy()   # 退出前调用
```

---

## 通信操作

### allreduce

```python
Comm_OP.allreduce(tensor, c_op="avg")
```

支持 `"sum"`、`"avg"` 计算方式，原地修改输入数据。

```python
from pyvqnet.distributed import CommController
from pyvqnet.tensor import tensor
import numpy as np

Comm_OP = CommController("gloo")

num = tensor.to_tensor(np.random.rand(1, 5))
print(f"rank {Comm_OP.getRank()}  {num}")

Comm_OP.allreduce(num, "sum")
print(f"rank {Comm_OP.getRank()}  {num}")
# vqnetrun --nproc_per_node 2 python test.py
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

将指定进程上的数据广播到所有进程。

```python
Comm_OP.broadcast(num, 1)
```

### allgather

```python
Comm_OP.allgather(tensor)
```

将所有进程上的数据 gather 到一起。

```python
num = Comm_OP.allgather(num)
```

### send / recv（P2P）

```python
Comm_OP.send(tensor, dest)
Comm_OP.recv(tensor, source)
```

点对点通信。

```python
from pyvqnet.distributed import CommController, get_rank
from pyvqnet.tensor import tensor
import numpy as np

Comm_OP = CommController("gloo")

num = tensor.to_tensor(np.random.rand(1, 5))
recv = tensor.zeros_like(num)

if get_rank() == 0:
    Comm_OP.send(num, 1)
elif get_rank() == 1:
    Comm_OP.recv(recv, 0)

print(f"rank {Comm_OP.getRank()}  {num}")
print(f"rank {Comm_OP.getRank()}  {recv}")
# vqnetrun --nproc_per_node 2 python test.py
```

---

## NCCL 异步操作

异步接口**仅在 NCCL（GPU）后端可用**（gloo 下请使用同步接口），均返回 `Work` 通信句柄，**必须**调用 `work.wait()` 等待操作完成后才能读取结果。

| 接口 | 签名 | 说明 |
|------|------|------|
| `nccl_async_all_gather` | `(output, input, group=None, async_op=False)` | 各进程 `input` 收集到 `output` |
| `nccl_async_all_reduce` | `(tensor, c_op="avg", group=None, async_op=False)` | 原地归约，`c_op` 取 `"sum"` / `"avg"` |
| `nccl_async_reduce` | `(tensor, dest, c_op="avg", group=None, async_op=False)` | 归约到 `dest` 进程，`c_op` 取 `"sum"` / `"avg"` |
| `nccl_async_broadcast` | `(tensor, src, group=None, async_op=False)` | 从 `src` 进程广播到所有进程 |
| `nccl_async_send` | `(tensor, dest, async_op=False)` | 点对点发送（无 `group` 参数） |
| `nccl_async_recv` | `(tensor, source, async_op=False)` | 点对点接收（无 `group` 参数） |

张量要求：

- 张量必须是 GPU 上的 QTensor：`.toGPU(pyvqnet.DEV_GPU_0 + get_local_rank())`；各进程 dtype、形状需一致
- `nccl_async_all_gather` 的 `output` 需预先分配，形状为 `[world_size, *input.shape]`，dtype 与 `input` 相同
- `nccl_async_send` / `nccl_async_recv` 的发送、接收张量 dtype 与形状必须匹配
- 支持 `kcomplex64`、`kfloat32` 等 QTensor dtype；`output` / `input` 的 dtype 必须一致
- 除 `nccl_async_send` / `nccl_async_recv` 外，其余接口都接受 `group` 参数；`group` 传 `split_groups` 返回的元素（`None` 表示全局组）
- `async_op=True` 在后台执行，`Work.wait()` 阻塞至完成；`async_op=False` 为同步语义，但仍返回 `Work`

示例（`nccl_async_all_reduce`）：

```python
import pyvqnet
from pyvqnet.tensor import tensor
from pyvqnet.distributed import CommController, get_local_rank

Comm_OP = CommController("nccl")

complex_data = tensor.ones([500, 500], dtype=pyvqnet.kcomplex64).toGPU(
    pyvqnet.DEV_GPU_0 + get_local_rank()
)
work = Comm_OP.nccl_async_all_reduce(complex_data, "sum", group=None, async_op=True)
work.wait()   # 异步操作：必须 wait 后才能使用结果

# vqnetrun --nproc_per_node 2 python test.py
```

> 异步接口只支持 NCCL；同一进程不要同时创建 gloo 和 NCCL 两个控制器。

---

## 进程组操作

### split_groups(rank_lists)

根据入参设置的进程号列表划分多个通信组。

**参数**：`rank_lists` —— 每个子列表是一个通信组的进程号，例如 `[[0, 1], [2, 3]]`。

**返回**：默认（pyvqnet-ad）后端返回元组列表，每个元素是一个通信组的进程号元组，长度等于分组个数；使用 torch 后端时返回 `torch.distributed.ProcessGroup` 列表。默认后端下本接口只登记组划分，真正创建通信组发生在首次用该组做组通信时。

```python
from pyvqnet.distributed import CommController, get_rank, get_local_rank
from pyvqnet.tensor import tensor
import numpy as np

Comm_OP = CommController("gloo")

groups = Comm_OP.split_groups([[0, 1], [2, 3]])
print(groups)   # 默认后端: [(0, 1), (2, 3)]

# vqnetrun --nproc_per_node 4 python test.py
```

### init_groups(rank_lists)

```python
pyvqnet.distributed.ControlComm.init_groups(rank_lists)
```

根据给出的进程号列表划分通信组。

**参数**：`rank_lists` —— 每个子列表是一个通信组的进程号，例如 `[[0, 2], [1]]`。

**返回**：默认（pyvqnet-ad）后端原样返回入参，即 `[[0, 2], [1]]` 这样的进程号列表；使用 torch 后端时返回 `torch.distributed.ProcessGroup` 列表。默认后端下本接口不创建通信组，组在首次用于组通信时创建。

```python
from pyvqnet.distributed import *
import numpy as np

Comm_OP = CommController("gloo")
num = tensor.to_tensor(np.random.rand(1, 5))
print(f"rank {Comm_OP.getRank()}  {num} before allreduce")

group_l = init_groups([[0, 2], [1]])

for comm_ in group_l:
    if Comm_OP.getRank() in comm_:
        Comm_OP.allreduce_group(num, "sum", group=comm_)
        print(f"rank {Comm_OP.getRank()}  {num} after allreduce")

# vqnetrun --nproc_per_node 3 python test.py
```

### 组内通信接口

```python
Comm_OP.allreduce_group(tensor, c_op="avg", group=None)
Comm_OP.reduce_group(tensor, root=0, c_op="avg", group=None)
Comm_OP.broadcast_group(tensor, root=0, group=None)
Comm_OP.allgather_group(tensor, group=None)
```

`group` 参数取 `split_groups` 返回值中的元素，例如 `(0, 1)`；也接受等价的列表 `[0, 1]`。默认 `None` 表示不分组。

> 默认后端下，`reduce_group` / `broadcast_group` 的 `root` 是**组内局部 rank**；torch 后端下为全局 rank。

```python
from pyvqnet.distributed import CommController, get_rank, get_local_rank
from pyvqnet.tensor import tensor
import numpy as np
from pyvqnet import kcomplex64

Comm_OP = CommController("nccl")
groups = Comm_OP.split_groups([[0, 1]])

complex_data = tensor.QTensor(
    [3 + 1j, 2, 1 + get_rank()], dtype=kcomplex64
).reshape((3, 1)).toGPU(1000 + get_local_rank())

print(f"allreduce_group before rank {get_rank()}: {complex_data}")
Comm_OP.allreduce_group(complex_data, c_op="sum", group=groups[0])
print(f"allreduce_group after rank {get_rank()}: {complex_data}")

# vqnetrun --nproc_per_node 2 python test.py
```

### allgather_group 示例

写法一：gloo 后端，CPU 张量。

```python
from pyvqnet.distributed import CommController, get_rank
from pyvqnet.tensor import tensor
from pyvqnet import kcomplex64

Comm_OP = CommController("gloo")
groups = Comm_OP.split_groups([[0, 1]])

complex_data = tensor.QTensor(
    [3 + 1j, 2, 1 + get_rank()], dtype=kcomplex64
).reshape((3, 1))
print(f"before rank {get_rank()}: {complex_data}")

complex_data = Comm_OP.allgather_group(complex_data, group=groups[0])
print(f"after rank {get_rank()}: {complex_data}")
# vqnetrun --nproc_per_node 2 python test.py
```

写法二：NCCL 后端，GPU 张量。两种写法二选一，不要在同一个进程里同时创建两个控制器。

```python
from pyvqnet.distributed import CommController, get_rank, get_local_rank
from pyvqnet.tensor import tensor
from pyvqnet import kcomplex64

Comm_OP = CommController("nccl")
groups = Comm_OP.split_groups([[0, 1]])

complex_data = tensor.QTensor(
    [3 + 1j, 2, 1 + get_rank()], dtype=kcomplex64
).reshape((3, 1)).toGPU(1000 + get_local_rank())
print(f"before rank {get_rank()}: {complex_data}")

complex_data = Comm_OP.allgather_group(complex_data, group=groups[0])
print(f"after rank {get_rank()}: {complex_data}")
# vqnetrun --nproc_per_node 2 python test.py
```

---

## 辅助函数

### split_data

在多进程中，使用 `split_data` 根据进程数对数据进行切分，返回相应进程上的数据。

> 以下示例需通过 `vqnetrun` 启动；直接 `python` 运行会因 `RANK` 未设置而报错。

```python
pyvqnet.distributed.datasplit.split_data(x_train, y_train, shuffle=False)
```

**参数**：

| 参数 | 说明 |
|------|------|
| `x_train` | `np.array` 训练数据 |
| `y_train` | `np.array` 训练数据标签 |
| `shuffle` | `bool` 是否打乱后再切分，默认 `False` |

**返回**：切分后的训练数据和标签。

> **切分规则**: `split_data` 按进程数均匀切分，并丢弃不能整除的尾部样本。例如数据量 n=5、进程数 size=2 时，rank0 得到 `[0,1]`、rank1 得到 `[2,3]`，样本 4 被丢弃。需要保留全部样本时请自行补齐长度或预先切分。

```python
from pyvqnet.distributed import split_data
import numpy as np

x_train = np.random.randint(255, size=(100, 5))
y_train = np.random.randint(2, size=(100, 1))

x_train, y_train = split_data(x_train, y_train)
```

### get_local_rank / get_rank / get_size

> 以下示例需通过 `vqnetrun` 启动；直接 `python` 运行会因 `RANK` 未设置而报错。

```python
from pyvqnet.distributed import get_local_rank, get_rank, get_size

get_local_rank()   # 当前机器上的进程号
get_rank()         # 全局进程号
get_size()         # 全局进程数
```

```python
from pyvqnet.distributed.ControlComm import get_rank

print(get_rank())
# vqnetrun --nproc_per_node 2 python test.py
```

> 在 torch 后端下使用 `get_local_rank()` 前，需要通过 `os.environ["LOCAL_RANK"] = str(rank)` 设置本地进程号。

### 环境变量

| 环境变量 | 说明 | 设置方式 |
|----------|------|----------|
| `LOCAL_RANK` | 本地进程编号 | `vqnetrun` 自动注入；torch 后端需手动设置 |
| `MASTER_ADDR` | 主节点地址 | `vqnetrun --master_addr` 注入 |
| `MASTER_PORT` | 主节点端口 | `vqnetrun --master_port` 注入 |
| `WORLD_SIZE` | 全局进程数 | `vqnetrun --nnodes × --nproc_per_node` |
| `NCCL_SOCKET_IFNAME` | NCCL 使用的网卡名 | `vqnetrun --nccl_socket_ifname` 注入 |

---

## 完整分布式训练示例

以下示例使用 gloo 后端在 2 个 CPU worker 上做手动梯度同步。

```python
from pyvqnet.distributed import CommController, get_rank, get_local_rank
from pyvqnet.nn import Module, Linear
from pyvqnet.optim import Adam
from pyvqnet.tensor import tensor
import numpy as np

# 初始化通信控制器: CPU 使用 gloo, GPU 使用 nccl
Comm_OP = CommController("gloo")
rank = get_rank()
local_rank = get_local_rank()

class Model(Module):
    def __init__(self):
        super().__init__()
        self.fc = Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

model = Model()
optimizer = Adam(model.parameters(), lr=0.01)

# 每个进程处理自己的数据分片
batch_size = 32
x = tensor.QTensor(np.random.randn(batch_size, 10).astype("float32"))
y = tensor.QTensor(np.random.randn(batch_size, 5).astype("float32"))
x.requires_grad = True

# 前向 / 反向
pred = model(x)
loss = ((pred - y) ** 2).mean()
optimizer.zero_grad()
loss.backward()

# 梯度同步（手动 allreduce）
for param in model.parameters():
    if param.grad is not None:
        Comm_OP.allreduce(param.grad, "avg")

optimizer.step()

if rank == 0:
    print(f"Loss: {loss.item()}")

Comm_OP.barrier()
Comm_OP.destroy()

# vqnetrun --nproc_per_node 2 python train.py
```

> GPU 版本将 `CommController("gloo")` 改为 `CommController("nccl")`，并把模型与数据通过 `.toGPU(get_local_rank() + DEV_GPU_0)` 移动到对应 GPU。
>
> `vqnetrun` 使用 spawn 启动子进程。实际脚本建议把初始化和训练逻辑放入 `def main():`，并在 `if __name__ == "__main__": main()` 中调用；`CommController` 应在每个 worker 进程内创建，避免父进程重复初始化。

---

## Pipeline Parallel（流水线并行）

Pipeline Parallel 将模型按层切分为多个 stage，每个 stage 分配到不同的 GPU 上执行，实现模型并行训练。采用 1F1B（One-Forward-One-Backward）调度策略，有效降低流水线气泡。仅在 Linux 且具有 GPU 时可用。

**何时使用**：模型过大无法放入单个 GPU 显存，或需要将模型的不同层分布在多 GPU 上执行。

### PipelineModule

将模型定义为一个层序列，自动分配到各 pipeline stage：

```python
from pyvqnet.distributed.pp.pp_module import PipelineModule, LayerSpec

PipelineModule(layers, num_stages=None, topology=None, loss_fn=None,
               base_seed=1234, partition_method="parameters")
```

**参数**：

| 参数 | 说明 |
|------|------|
| `layers` | 层序列（`nn.Sequential` 或层列表） |
| `num_stages` | pipeline stage 数量 |
| `topology` | `PipeDataParallelTopology`，指定 pipeline 和数据并行拓扑 |
| `loss_fn` | 损失函数 |
| `partition_method` | 层划分方法：`"uniform"`（均匀）、`"parameters"`（按参数量）、`"type:<name>"`（按类型） |

### LayerSpec

用于延迟构建的层规格说明：

```python
LayerSpec(typename, *module_args, **module_kwargs)
```

### PipelineParallelTrainingWrapper

高级训练封装，自动管理 engine/optimizer/dataloader：

```python
from pyvqnet.distributed.pp import PipelineParallelTrainingWrapper

PipelineParallelTrainingWrapper(args, join_layers, trainset)
```

**参数说明 (args dict)**：

| 键 | 说明 |
|----|------|
| `"backend"` | `"nccl"`（仅 GPU 支持） |
| `"train_batch_size"` | 全局 batch size |
| `"train_micro_batch_size_per_gpu"` | 每 GPU micro batch size |
| `"epochs"` | 训练轮数 |
| `"optimizer"` | 优化器配置 `{"type": "Adam", "params": {"lr": 0.001}}` |
| `"pipeline_parallel_size"` | pipeline stage 数量 |
| `"loss"` | 损失函数实例 |
| `"steps"` | 每 epoch 步数 |
| `"partition_method"` | （可选）层划分方法 |

**示例**：

```python
import os
import pyvqnet.nn as nn
from pyvqnet.nn import Sequential, CrossEntropyLoss, Linear, Conv2D, MaxPool2D, activation as F
from pyvqnet.distributed.pp import PipelineParallelTrainingWrapper
from pyvqnet.distributed.configs import comm as dist
from pyvqnet.tensor import tensor

pipeline_parallel_size = 2
num_steps = 1000

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Sequential(Conv2D(3, 8, 3, padding='same'), F.ReLu(), MaxPool2D([2, 2], [2, 2]))
        self.cls = Sequential(Linear(8 * 112 * 112, 10))   # 224x224 输入经一次 MaxPool2D 后为 8x112x112

    def forward(self, x):
        return self.cls(tensor.flatten(self.features(x), 1))

def join_layers(model):
    return [*model.features, lambda x: tensor.flatten(x, 1), *model.cls]

if __name__ == "__main__":
    os.environ["LOCAL_RANK"] = str(dist.get_local_rank())

    args = {
        "backend": "nccl",
        "train_batch_size": 64,
        "train_micro_batch_size_per_gpu": 32,
        "epochs": 5,
        "optimizer": {"type": "Adam", "params": {"lr": 0.001}},
        "local_rank": dist.get_local_rank(),
        "pipeline_parallel_size": pipeline_parallel_size,
        "loss": CrossEntropyLoss(),
        "steps": num_steps,
        "seed": 42,
    }

    trainset = ...  # 数据集
    w = PipelineParallelTrainingWrapper(args, join_layers(Model()), trainset)

    for i in range(args["epochs"]):
        w.train_batch()

    all_loss = w.loss_dict

# vqnetrun --nproc_per_node 2 python train.py
```

**注意事项**：

- 仅支持 Linux + GPU（NCCL）环境
- 每个进程需要在 `__main__` 中配置环境变量 `LOCAL_RANK`
- 模型需表示为层序列，`join_layers` 函数将模型展平为层列表
- 每个 stage 自动分配到对应 GPU
- 启动方式：`vqnetrun --nproc_per_node 2 python train.py`

---

## ParallelTrainingWrapper（混合并行）

`ParallelTrainingWrapper` 是统一的并行训练封装器，支持三种并行模式：

- 经典层流水线并行（PP）
- 数据并行（DP）
- 比特并行（Bit）

它可将经典层与变分量子线路（VQC）混合编排到多级流水线中执行。

```python
pyvqnet.distributed.hybrid.ParallelTrainingWrapper(args, pipeline_def, trainset, strategy=None)
```

### args 配置字典

| 键 | 类型 / 默认值 | 说明 |
|----|---------------|------|
| `data_parallel_size` | int, 默认 1 | 数据并行度 |
| `pipeline_parallel_size` | int, 必须等于 stage 总数 | 流水线并行度 |
| `qubits_parallel_size` | int, 默认 1 | 比特并行度 |
| `train_batch_size` | int | 全局 batch 大小 |
| `train_micro_batch_size_per_gpu` | int | 每个 micro batch 大小 |
| `epochs` | int, 默认 1 | 训练轮数 |
| `steps` | int, 默认 10 | 每个 batch 的训练步数 |
| `loss` | loss function | 损失函数，仅在最后一个 stage 需要 |
| `optimizer` | dict, 默认 `{"type": "Adam", "params": {"lr": 0.001}}` | 优化器配置 |
| `device` | `"cpu"` / `"gpu"`，省略则自动检测 | 计算设备 |
| `steps_per_print` | int, 默认 1 | 打印间隔步数 |
| `seed` | int, 默认 42 | 随机种子，用于 VQC 录制的确定性 |

### pipeline_def 的四种形式

1. **手写字典列表（旧版 pipeline_def）**：

   ```python
   [{"type": "classical", "layers": [...]}, ...]
   ```

   每个条目是一个 stage 字典，支持 `classical`、`vqc_split`、`vqc_whole` 三种类型。

2. **纯经典模型**：`nn.Module` 层列表，自动分配到各 stage。

3. **纯 VQC 模型**：一个 `QModule` 子类（必须是类，不能是实例）。

4. **混合模型**：`("classical", [layers])` / `("vqc", VQCClass)` 元组列表。

形式 2–4 通过 `auto_plan` 自动转换为旧版字典列表中间表示。

### strategy 策略字典

`strategy` 仅用于声明式模型定义（形式 2–4），可选键：

| 键 | 取值 | 说明 |
|----|------|------|
| `classical_split` | `"uniform"` / `"auto"`（默认 `"auto"`） | 经典层分配策略 |
| `vqc_split` | `None` / `False`（默认 `None`） | `None` 表示自动切分 VQC 块；`False` 表示保持每个 VQC 块完整 |
| `compression_policy` | 默认 `"greedy"` | stage 合并策略 |

### 约束与启动

- `pipeline_parallel_size` 必须等于 `pipeline_def` 中 stage 的总数。
- 必须通过 `vqnetrun --nproc_per_node N` 启动。
- `data_parallel_size × pipeline_parallel_size × qubits_parallel_size` **必须等于总进程数**。
- 仅支持 Linux。

### train_batch()

执行一个 batch 的完整训练流程（前向 → 反向 → 参数更新）。

### 最小结构示例

以下示例给出 2-stage 复合流水线（classical + VQC | VQC + classical）的配置结构。`MyVQC` 为包含 `HybridDistributedQMachine` 的 VQC 模块。完整可运行示例见 [VQNet 分布式计算模块官方文档](https://qcloud.originqc.com.cn/document/vqnet_api_cn/index.html#) 的 ParallelTrainingWrapper 章节。

```python
import os
import pyvqnet.nn as nn
from pyvqnet.nn import CrossEntropyLoss
from pyvqnet.distributed.configs import comm as dist
from pyvqnet.distributed.hybrid import ParallelTrainingWrapper

NUM_WIRES = 8

# MyVQC 必须是包含 HybridDistributedQMachine 的 VQC 模块；完整定义见官方示例
class MyVQC(nn.Module):
    ...

def main():
    os.environ["LOCAL_RANK"] = str(dist.get_local_rank())

    args = {
        "device": "cpu",
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 32,
        "epochs": 1,
        "steps": 1,
        "data_parallel_size": 1,
        "pipeline_parallel_size": 2,
        "qubits_parallel_size": 2,
        "loss": CrossEntropyLoss(),
        "optimizer": {"type": "Adam", "params": {"lr": 0.01}},
        "steps_per_print": 50,
        "seed": 42,
    }

    # 2-stage 复合流水线：classical + VQC | VQC + classical
    pipeline_def = [
        {"type": "stage", "sub": [
            {"type": "classical", "layers": [nn.Linear(7, NUM_WIRES)]},
            {"type": "vqc_split", "module_class": MyVQC, "vqc_stages": 2, "vqc_slice": 0},
        ]},
        {"type": "stage", "sub": [
            {"type": "vqc_split", "module_class": MyVQC, "vqc_stages": 2, "vqc_slice": 1},
            {"type": "classical", "layers": [nn.Linear(NUM_WIRES, 8)]},
        ]},
    ]

    trainset = ...  # 需具有 .data / .labels 且 __getitem__ 返回 (data, label)
    w = ParallelTrainingWrapper(args, pipeline_def, trainset)
    w.train_batch()

if __name__ == "__main__":
    main()

# vqnetrun --nproc_per_node 4 python train.py
```

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

- `tp_comm` —— 必须传入 `CommController` 实例用于通信

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
from pyvqnet.device import DEV_GPU_0

pyvqnet.utils.set_random_seed(42)

Comm_OP = CommController("nccl")
local_rank = Comm_OP.get_local_rank()

fc1 = ColumnParallelLinear(8, 6, tp_comm=Comm_OP)
fc2 = RowParallelLinear(6, 4, tp_comm=Comm_OP)

fc1 = fc1.toGPU(local_rank + DEV_GPU_0)
fc2 = fc2.toGPU(local_rank + DEV_GPU_0)

z = pyvqnet.tensor.ones([2, 8]).toGPU(local_rank + DEV_GPU_0)
z.requires_grad = True

y = fc2(fc1(z))
y.backward()

Comm_OP.barrier()
Comm_OP.destroy()

# vqnetrun --nproc_per_node 2 python test.py
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

**参数**：

| 参数 | 说明 |
|------|------|
| `args` | 参数字典，见下文 |
| `model` | 输入模型 |
| `optimizer` | 优化器；为 `None` 时采用 `args` 中 `optimizer` 的设置 |

**参数说明 (args dict)**：

| 键 | 说明 |
|----|------|
| `"train_batch_size"` | 训练 batch size（等效的总 batch） |
| `"optimizer"` | 优化器配置 `{"type": "adam", "params": {"lr": 0.001}}` |
| `"zero_optimization"` | ZeRO 配置 `{"stage": 1}` 或 `{"stage": 2}` |

**使用流程**：`model.backward(loss)` → `model.step()`（基于返回的 model 做反向与更新）。

**示例 (Stage 1)**：

```python
import os
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
        self.net = nn.Sequential(nn.Linear(784, 256), nn.activation.ReLu(),
                                 nn.Linear(256, 128), nn.activation.ReLu(),
                                 nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x.reshape([-1, 784]))

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

os.environ["LOCAL_RANK"] = str(get_local_rank())
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

Comm_OP.destroy()

# vqnetrun --nproc_per_node 2 python test.py
```

---

## DataParallelVQCLayer（VQC 数据并行层）

提供 VQC（变分量子线路）的数据并行封装，自动处理梯度同步。支持普通模式和 adjoint 模式。

### DataParallelVQCLayer

标准 VQC 数据并行层：

```python
from pyvqnet.distributed import DataParallelVQCLayer

DataParallelVQCLayer(Comm_OP, vqc_module, name="",
                     use_bucket_cpp_allreduce=True,
                     bucket_bytes_cap=25 * 1024 * 1024,
                     first_bucket_bytes_cap=1024 * 1024,
                     find_unused_parameters=False,
                     gradient_as_bucket_view=False)
```

**参数**：

| 参数 | 说明 |
|------|------|
| `Comm_OP` | `CommController` 实例 |
| `vqc_module` | 包含 `forward()` 的 VQC Module（需正确设置 QMachine） |
| `name` | 模块名称，默认空字符串 |
| `use_bucket_cpp_allreduce` | 是否使用 C++ 分桶 all-reduce 实现。**当前必须为 `True`**；传 `False` 会抛出 `NotImplementedError`（旧的逐参数 all-reduce 已移除）。默认 `True` |
| `bucket_bytes_cap` | 梯度桶的最大字节数，默认 `25*1024*1024` |
| `first_bucket_bytes_cap` | 第一个梯度桶的最大字节数，默认 `1024*1024` |
| `find_unused_parameters` | 是否查找 autograd 图中未使用的参数，默认 `False` |
| `gradient_as_bucket_view` | 是否将梯度暴露为通信桶的视图，默认 `False` |

### DataParallelVQCAdjointLayer

使用 adjoint 梯度的 VQC 数据并行层（继承自 `QuantumLayerAdjoint`），构造参数与 `DataParallelVQCLayer` **完全相同**。用法同上面的 `DataParallelVQCLayer` 示例，只需：

- 将 `qmodel` 的 `QMachine` 设为 `grad_mode="adjoint"`（如 `QModel(..., grad_mode="adjoint")`）
- 从 `pyvqnet.distributed` 导入 `DataParallelVQCAdjointLayer`，用 `layer = DataParallelVQCAdjointLayer(Comm_OP, qmodel)` 替换原层并调用 `layer.train()`；数据分片由用户自行完成，每个节点处理自己的分片

### DataParallelLayer

通用数据并行层（导入路径）：

```python
from pyvqnet.distributed import DataParallelLayer
```

### C++ 分桶 Allreduce

默认启用 C++ 分桶 all-reduce。分桶路径下可在 forward/backward 前后调用：

```python
layer = DataParallelVQCLayer(Comm_OP, vqc_module)   # use_bucket_cpp_allreduce 必须为 True
y = layer(x)                        # forward
layer.prepare_for_backward(y)       # backward 前准备
y.backward()
layer.finalize_backward()           # backward 后、优化器 step 前结束
```

### DataParallelVQCLayer 示例

```python
# vqnetrun --nproc_per_node 2 python test.py

import pyvqnet
from pyvqnet.qnn.vqc import QMachine, cnot, rx, rz, MeasureAll, PauliX
from pyvqnet.tensor import tensor
from pyvqnet.nn import Module
from pyvqnet.distributed import CommController, DataParallelVQCLayer
from pyvqnet.device import DEV_GPU_0

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

Comm_OP = CommController("nccl")

input = tensor.ones([b, 3 * n * l]).toGPU(Comm_OP.get_local_rank() + DEV_GPU_0)
input.requires_grad = True

qmodel = QModel(num_wires=n, num_layer=l, dtype=pyvqnet.kcomplex64)
qmodel.toGPU(Comm_OP.get_local_rank() + DEV_GPU_0)

layer = DataParallelVQCLayer(Comm_OP, qmodel)
y = layer(input)
y.backward()

Comm_OP.barrier()
Comm_OP.destroy()
```

---

## Qubits Reorder（量子比特重排）

Qubits Reorder 模块提供跨设备的量子比特分布执行，支持将大规模量子线路拆分到多个计算节点上运行。其核心是通过改变比特并行过程中量子逻辑门的排列顺序，减少比特并行中需要执行比特变换的次数。参考论文 [Lazy Qubit Reordering for Accelerating Parallel State-Vector-based Quantum Circuit Simulation](https://export.arxiv.org/abs/2410.04252)。

### 导入路径

```python
from pyvqnet.distributed.qubits_reorder import (
    DistributedQMachine, DistQuantumLayerAdjoint, QubitReorder,
)
# 同一导入路径下的其他接口: QubitReorderOp, QubitsPermutation,
# compute_qubit_mapping, compute_qubit_remapping, flat_time_space_tiling,
# gen_qr_machine, swap_global_global_q1, swap_local_global_q1,
# TorchDistributedQMachine, TorchDistQuantumLayerAdjoint,
# gen_local_qm_and_op_history, measure_all_for_qr, exec_states_qr,
# update_global_qmachine_states_in_qr, update_local_states_with_qr
```

### 核心理念

Qubits Reorder 通过以下机制实现跨设备量子电路执行：

1. **量子比特映射**：将逻辑量子比特映射到物理设备
2. **时空平铺**：`flat_time_space_tiling` 将量子比特在时间-空间维度上展开
3. **SWAP 网络**：通过 `swap_global_global_q1` / `swap_local_global_q1` 在设备间交换量子比特
4. **分布式期望值**：跨设备计算期望值

### DistributedQMachine

```python
pyvqnet.distributed.qubits_reorder.DistributedQMachine(num_wires, dtype=kcomplex64, grad_mode="")
```

用于比特并行中的变分量子计算的模拟类，包含每个节点上的部分比特量子态。每个节点各自创建一个该类的实例，通过 gloo / NCCL 进行分布式的量子变分线路模拟。进程数 N 必须等于 2 的 `global_qubit` 次方，可通过 `set_qr_config` 进行配置。

**参数**：

| 参数 | 说明 |
|------|------|
| `num_wires` | 完整量子线路的比特数量 |
| `dtype` | 计算数据的数据类型，默认 `pyvqnet.kcomplex64`，对应参数精度 `pyvqnet.kfloat32` |
| `grad_mode` | 在 `DistQuantumLayerAdjoint` 进行反传时设置为 `"adjoint"` |

> **注意**：输入的比特数是整个量子线路所需要的比特数量，`DistributedQMachine` 会根据全局比特数构建量子模拟器，其比特数量为 `num_wires - global_qubit`。反传必须基于 `DistQuantumLayerAdjoint`。

> **警告**：该接口只支持在 Linux 下运行。必须对 `DistributedQMachine` 中比特并行参数进行配置：

```python
qm.set_just_defined(True)
qm.set_save_op_history_flag(True)   # open save op
qm.set_qr_config({'qubit': 总比特个数, 'global_qubit': 分布式比特个数})
```

**示例**：

```python
from pyvqnet.distributed import get_rank
from pyvqnet import tensor
from pyvqnet.qnn.vqc import rx, ry, cnot, MeasureAll, rz
import pyvqnet
from pyvqnet.distributed.qubits_reorder import DistributedQMachine, DistQuantumLayerAdjoint

pyvqnet.utils.set_random_seed(123)

qubit = 10

class QModel(pyvqnet.nn.Module):
    def __init__(self, num_wires, dtype, grad_mode=""):
        super(QModel, self).__init__()
        self.qm = DistributedQMachine(num_wires, dtype=dtype, grad_mode=grad_mode)
        self.qm.set_just_defined(True)
        self.qm.set_save_op_history_flag(True)
        self.qm.set_qr_config({"qubit": num_wires, "global_qubit": 1})
        self.params = pyvqnet.nn.Parameter([qubit])
        self.measure = MeasureAll(obs={"X5": 1.0})

    def forward(self, x, *args, **kwargs):
        self.qm.reset_states(x.shape[0])
        for i in range(qubit):
            rx(q_machine=self.qm, params=self.params[i], wires=i)
            ry(q_machine=self.qm, params=self.params[3], wires=i)
            rz(q_machine=self.qm, params=self.params[4], wires=i)
        cnot(q_machine=self.qm, wires=[0, 1])
        return self.measure(q_machine=self.qm)

input_x = tensor.QTensor([[0.1, 0.2, 0.3]], requires_grad=True).toGPU(
    pyvqnet.DEV_GPU_0 + get_rank()
)
input_x = tensor.broadcast_to(input_x, [2, 3])
input_x.requires_grad = True

quantum_model = QModel(
    num_wires=qubit, dtype=pyvqnet.kcomplex64, grad_mode="adjoint"
).toGPU(pyvqnet.DEV_GPU_0 + get_rank())

adjoint_model = DistQuantumLayerAdjoint(quantum_model)
adjoint_model.train()

batch_y = adjoint_model(input_x)
batch_y.backward()
print(batch_y)

# vqnetrun --nproc_per_node 2 python test.py
```

### DistQuantumLayerAdjoint

```python
pyvqnet.distributed.qubits_reorder.DistQuantumLayerAdjoint(vqc_module, name="")
```

使用伴随矩阵方式对比特并行计算中的参数进行梯度计算的 `DistQuantumLayer` 层。

**参数**：

| 参数 | 说明 |
|------|------|
| `vqc_module` | 输入的蕴含 `DistributedQMachine` 的模块 |
| `name` | 模块名称，默认空字符串 |

> **注意**：输入的 `vqc_module` 必须包含 `DistributedQMachine`，基于 `DistributedQMachine` 进行比特并行下的 adjoint 反传梯度计算。
>
> **警告**：该接口只支持在 Linux 下运行。

```python
from pyvqnet.distributed.qubits_reorder import DistQuantumLayerAdjoint

adjoint_model = DistQuantumLayerAdjoint(quantum_model, name="dist_vqc")
adjoint_model.train()
batch_y = adjoint_model(input_x)
batch_y.backward()
```

---

## 高级函数参考

### broadcast_model_params

```python
Comm_OP.broadcast_model_params(model, root=0)   # 从指定进程广播模型参数
```

### 梯度相关接口

| 接口 | 说明 |
|------|------|
| `Comm_OP.grad_allreduce(optimizer)` | 对所有参数的梯度做 all-reduce |
| `Comm_OP.param_allreduce(model)` | 对所有参数做 all-reduce |
| `post_grad_all_reduce(all_parameters)` | 为所有参数创建梯度并做 all-reduce |

```python
from pyvqnet.distributed import post_grad_all_reduce

post_grad_all_reduce(model.parameters())
```

---

## 常见问题

### Q: 分布式功能支持哪些系统？

**仅支持 Linux**。其他平台会抛出 “VQNet distributed features are only supported on Linux platform” 错误。

### Q: CPU 和 GPU 分布式分别使用什么通信后端？

- **CPU**：`CommController("gloo")`
- **GPU**：`CommController("nccl")`
- 不传 backend 时，`CommController()` 根据通信数据所在设备自动选择

### Q: 多节点运行时需要注意什么？

- 各节点使用相同的 Python 环境与 VQNet 版本
- 各节点独立执行 `vqnetrun`，通过 `--nnodes`、`--node_rank`、`--master_addr`、`--master_port` 完成会合
- 多网卡环境下可用 `--nccl_socket_ifname` 指定 NCCL 通信网卡
- `WORLD_SIZE = nnodes * nproc_per_node`，所有节点该值必须一致

---

**Version**: VQNet 2.18.2
