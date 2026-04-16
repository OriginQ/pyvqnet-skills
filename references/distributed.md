# VQNet 分布式计算模块 API Reference

> 来源: VQNET2.0-tutorial/source/rst/vqnet_dist.rst
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

### 进程数 (-n/-np)

```bash
vqnetrun -n 2 python train.py
vqnetrun -np 4 python train.py
```

### 节点指定 (-H/--hosts)

```bash
vqnetrun -np 4 -H node0:1,node2:1 python train.py
vqnetrun -np 4 --hosts node0:1,node2:1 python train.py
```

### Hostfile (-f/-hostfile/--hostfile)

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

### 输出保存 (--output-filename)

```bash
vqnetrun -np 4 --hostfile hosts --output-filename output python train.py
```

### 详细输出 (--verbose)

```bash
vqnetrun -np 4 --hostfile hosts --verbose python train.py
```

### 超时设置 (--start-timeout)

默认 30 秒。

```bash
vqnetrun -np 4 --start-timeout 10 python train.py
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

### split_group

```python
Comm_OP.split_group(rankL)
```

划分多个通信组。

```python
from pyvqnet.distributed import CommController

Comm_OP = CommController("mpi")
groups = Comm_OP.split_group([[0, 1], [2, 3]])
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
groups = Comm_OP.split_group([[0, 1]])

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
**Source**: VQNET2.0-tutorial/source/rst/vqnet_dist.rst