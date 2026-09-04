# Classical Neural Network API Reference

> **重要**: 所有示例代码均来自官方文档，可直接运行。

---

## Module 基类

```python
pyvqnet.nn.module.Module(name="")
```

所有神经网络模块的基类。

**关键方法**:
- `forward(x, *args, **kwargs)` - 前向传播，必须被子类实现
- `parameters()` - 返回所有可训练参数
- `named_parameters()` - 返回参数名和参数
- `toGPU(device=DEV_GPU_0)` - 移动模型到 GPU
- `train()` / `eval()` - 设置训练/评估模式

**示例**:
```python
from pyvqnet.nn import Module, Linear, ReLU

class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 8)
        self.relu = ReLU()
        self.fc2 = Linear(8, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP()
model.train()
print(model.parameters())
```

---

## ModuleList / Sequential

### ModuleList

```python
pyvqnet.nn.ModuleList(modules=None)
```

用于存储子模块列表（支持索引访问）。

```python
from pyvqnet.nn import Module, ModuleList, Linear

class MyModel(Module):
    def __init__(self):
        super().__init__()
        self.layers = ModuleList([Linear(10, 20) for _ in range(3)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

### Sequential

```python
pyvqnet.nn.Sequential(*args)
```

顺序容器，自动执行前向传播。

```python
from pyvqnet.nn import Sequential, Linear, ReLU

model = Sequential(
    Linear(4, 8),
    ReLU(),
    Linear(8, 2)
)

output = model(input)
```

---

## Linear 全连接层

```python
pyvqnet.nn.Linear(input_channels, output_channels, weight_initializer=None, bias_initializer=None, use_bias=True, dtype=None, name="")
```

**公式**: `y = x @ A.T + b`

**参数**:
- `input_channels` - 输入特征维度
- `output_channels` - 输出特征维度
- `use_bias` - 是否使用偏置，默认 True

**示例**:
```python
import numpy as np
import pyvqnet
from pyvqnet.tensor import QTensor
from pyvqnet.nn import Linear

n = Linear(7, 5)
input = QTensor(np.arange(1, 43).reshape((2, 3, 7)), requires_grad=True, dtype=pyvqnet.kfloat32)
y = n.forward(input)
print(y)
```

---

## Conv2D 卷积层

```python
pyvqnet.nn.Conv2D(input_channels, output_channels, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, use_bias=True, weight_initializer=None, bias_initializer=None, dtype=None, name="")
```

2D 卷积层，输入形状 `(B, C_in, H, W)`。

**参数**:
- `input_channels` - 输入通道数
- `output_channels` - 输出通道数
- `kernel_size` - 卷积核大小 (int 或 tuple)
- `stride` - 步长，默认 (1, 1)
- `padding` - 填充，默认 (0, 0)
- `dilation` - 空洞卷积参数
- `groups` - 分组卷积

**示例**:
```python
import numpy as np
from pyvqnet.tensor import QTensor
from pyvqnet.nn import Conv2D
import pyvqnet

b, ic, oc = 2, 2, 3
test_conv = Conv2D(ic, oc, (2, 2), (2, 2), (0, 0))
x = QTensor(np.arange(1, 17).reshape([b, ic, 4, 1]), requires_grad=True, dtype=pyvqnet.kfloat32)
y = test_conv.forward(x)
print(y)
```

---

## Conv1D 卷积层

```python
pyvqnet.nn.Conv1D(input_channels, output_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bias=True, weight_initializer=None, bias_initializer=None, dtype=None, name="")
```

1D 卷积层，输入形状 `(B, C_in, L)`。

---

## ConvT2D 转置卷积层

```python
pyvqnet.nn.ConvT2D(input_channels, output_channels, kernel_size, stride=(1, 1), padding="valid", use_bias=True, dilation_rate=(1, 1), group=1, kernel_initializer=None, bias_initializer=None, dtype=None, name="")
```

转置卷积（反卷积）层，用于上采样，输入形状 `(B, C_in, H, W)`。

**参数**:
- `input_channels` - 输入通道数
- `output_channels` - 输出通道数
- `kernel_size` - 卷积核大小 (int 或 tuple)
- `stride` - 步长，默认 (1, 1)
- `padding` - `"valid"`（无填充）或 `"same"`（输出与输入同尺寸）
- `dilation_rate` - 空洞卷积参数
- `group` - 分组卷积

```python
from pyvqnet.nn import ConvT2D
from pyvqnet.tensor import QTensor
from pyvqnet.utils import initializer
import numpy as np

test_conv = ConvT2D(3, 2, [3, 3], [1, 1], "valid", True, initializer.ones, initializer.ones)
x = QTensor(np.arange(1, 76).reshape([1, 3, 5, 5]), requires_grad=True)
y = test_conv.forward(x)
print(y)
```

---

## BatchNorm 归一化层

### BatchNorm2d

```python
pyvqnet.nn.BatchNorm2d(channel_num, momentum=0.1, epsilon=1e-5, affine=True, beta_initializer=zeros, gamma_initializer=ones, dtype=None, name="")
```

4D 输入批归一化 `(B, C, H, W)`。

**示例**:
```python
import numpy as np
from pyvqnet.tensor import QTensor
from pyvqnet.nn import BatchNorm2d
import pyvqnet

test_conv = BatchNorm2d(2)
x = QTensor(np.arange(1, 17).reshape([2, 2, 4, 1]), requires_grad=True, dtype=pyvqnet.kfloat32)
y = test_conv.forward(x)
print(y)
```

### BatchNorm1d

```python
pyvqnet.nn.BatchNorm1d(channel_num, momentum=0.1, epsilon=1e-5, affine=True, dtype=None, name="")
```

2D 输入批归一化 `(B, C)`。

---

## LayerNorm 层归一化

### LayerNorm1d

```python
pyvqnet.nn.layer_norm.LayerNorm1d(norm_size, epsilon=1e-5, affine=True, dtype=None, name="")
```

**示例**:
```python
import numpy as np
import pyvqnet
from pyvqnet.tensor import QTensor
from pyvqnet.nn.layer_norm import LayerNorm1d

test_conv = LayerNorm1d(4)
x = QTensor(np.arange(1, 17).reshape([4, 4]), requires_grad=True, dtype=pyvqnet.kfloat32)
y = test_conv.forward(x)
print(y)
```

### LayerNorm2d / LayerNormNd

```python
pyvqnet.nn.layer_norm.LayerNorm2d(norm_size, epsilon=1e-5, affine=True, dtype=None, name="")
pyvqnet.nn.layer_norm.LayerNormNd(normalized_shape, epsilon=1e-5, affine=True, dtype=None, name="")
```

---

## GroupNorm

```python
pyvqnet.nn.GroupNorm(num_groups, num_channels, epsilon=1e-5, affine=True, dtype=None, name="")
```

分组归一化。

```python
from pyvqnet.nn import GroupNorm
test_conv = GroupNorm(2, 10)
x = QTensor(np.arange(0, 60*2*5).reshape([2, 10, 3, 2, 5]), requires_grad=True, dtype=kfloat32)
y = test_conv.forward(x)
```

---

## RMSNorm

```python
pyvqnet.nn.RMSNorm(normalized_shape, eps=1e-5, affine=True, dtype=None)
```

Root Mean Square Layer Normalization。相比 LayerNorm 不进行均值中心化（无 bias），计算更轻量。

**公式**: `y = x / RMS(x) * γ`, 其中 `RMS(x) = sqrt(ε + mean(x²))`

**参数**:
- `normalized_shape` - 归一化形状（int 或 tuple），作用于最后一维
- `eps` - 数值稳定性常数，默认 1e-5
- `affine` - 是否使用可学习的缩放参数 γ，默认 True

```python
import numpy as np
from pyvqnet.tensor import QTensor, kfloat32
from pyvqnet.nn import RMSNorm

rms = RMSNorm(4)
x = QTensor(np.arange(1, 9).reshape([2, 4]), requires_grad=True, dtype=kfloat32)
y = rms(x)
print(y)
```

---

## Pooling 池化层

### MaxPool2D

```python
pyvqnet.nn.MaxPool2D(kernel_shape=(2, 2), stride=(2, 2), padding=(0, 0), name="")
```

2D 最大池化。

### AvgPool2D

```python
pyvqnet.nn.AvgPool2D(kernel_shape=(2, 2), stride=(2, 2), padding=(0, 0), name="")
```

2D 平均池化。

### AdaptiveAvgPool2d

```python
pyvqnet.nn.AdaptiveAvgPool2d(out_size, name="")
```

自适应平均池化，输出尺寸固定为 `out_size`，不依赖输入尺寸。

```python
from pyvqnet.nn import AdaptiveAvgPool2d
from pyvqnet.tensor import QTensor
import numpy as np

layer = AdaptiveAvgPool2d((1, 1))
x = QTensor(np.arange(16).reshape([1, 1, 4, 4]), requires_grad=True, dtype=pyvqnet.kfloat32)
y = layer(x)
print(y.shape)  # (1, 1, 1, 1)
```

### MaxPool1D

```python
pyvqnet.nn.MaxPool1D(kernel, stride, padding="valid", name="")
```

1D 最大池化。

```python
from pyvqnet.nn import MaxPool1D
from pyvqnet.tensor import QTensor
import numpy as np

test_mp = MaxPool1D([3], [2], "same")
x = QTensor(np.array([0, 1, 0, 4, 5, 2, 3, 2, 1, 3], dtype=float).reshape([1, 1, 10]), requires_grad=True)
y = test_mp(x)
print(y)
```

### AvgPool1D

```python
pyvqnet.nn.AvgPool1D(kernel, stride, padding="valid", name="")
```

1D 平均池化。

---

## Dropout

```python
pyvqnet.nn.Dropout(dropout_rate=0.5)
```

随机丢弃神经元。

### DropPath

```python
pyvqnet.nn.DropPath(dropout_rate=0.5, name="")
```

Drop Path（随机深度），在残差网络训练中按样本随机丢弃整个残差路径，用于正则化深层网络。

```python
from pyvqnet.nn import DropPath
from pyvqnet.tensor import tensor

x = tensor.randu([4, 16, 16])
layer = DropPath(0.2)
layer.train()
y = layer(x)
```

```python
from pyvqnet.nn.dropout import Dropout
from pyvqnet.tensor import QTensor
import numpy as np

droplayer = Dropout(0.5)
droplayer.train()
x = QTensor(np.arange(-8, 8).reshape([2, 2, 2, 2]), requires_grad=True)
y = droplayer(x)
print(y)
```

---

## Embedding

```python
pyvqnet.nn.Embedding(num_embeddings, embedding_dim, weight_initializer=None, dtype=None, name="")
```

词嵌入层。

**注意**: 输入必须是 `kint64` 类型。

```python
from pyvqnet.nn import Embedding
from pyvqnet.tensor import QTensor
import pyvqnet

emb = Embedding(5, 3)
input_data = QTensor([[1, 2, 3], [1, 3, 4]], dtype=pyvqnet.kint64, requires_grad=False)
output = emb(input_data)
print(output)
```

---

## RNN / LSTM / GRU

### LSTM

```python
pyvqnet.nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name="")
```

长短期记忆网络。

```python
from pyvqnet.nn import LSTM
from pyvqnet.tensor import tensor

rnn2 = LSTM(4, 6, 2, batch_first=False, bidirectional=True)
input = tensor.ones([5, 3, 4])
h0 = tensor.ones([4, 3, 6])
c0 = tensor.ones([4, 3, 6])
output, (hn, cn) = rnn2(input, (h0, c0))
print(output)
```

### GRU

```python
pyvqnet.nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name="")
```

门控循环单元。

### RNN

```python
pyvqnet.nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name="")
```

循环神经网络。

### Dynamic_RNN / Dynamic_LSTM / Dynamic_GRU

支持变长序列输入的动态 RNN，配合 `tensor.PackedSequence` 使用。需通过 `pad_sequence`、`pack_pad_sequence` 构建打包序列。

```python
pyvqnet.nn.Dynamic_RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name="")
pyvqnet.nn.Dynamic_LSTM(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name="")
pyvqnet.nn.Dynamic_GRU(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name="")
```

---

## RoPE 旋转位置编码

```python
pyvqnet.nn.RoPE(head_dim, max_seq_len=2048, base=10000.0, rope_type="standard", scale_factor=4.0, original_max_seq_len=None, beta_fast=32.0, beta_slow=1.0)
```

旋转位置编码（Rotary Position Embedding），支持 4 种变体：

**参数**:
- `head_dim` - 注意力头维度（必须为偶数）
- `max_seq_len` - 预计算 cos/sin 缓存的最大序列长度
- `base` - RoPE 基础频率（Llama 使用 500000.0）
- `rope_type` - 变体类型：
  - `"standard"` - 标准 RoPE
  - `"ntk"` - NTK-aware RoPE（扩展上下文）
  - `"dynamic_ntk"` - 动态 NTK（根据实际序列长度动态调整）
  - `"yarn"` - YaRN（插值 + NTK 混合）
- `scale_factor` - 上下文扩展倍数
- `original_max_seq_len` - 预训练序列长度
- `beta_fast` / `beta_slow` - YaRN 频率截止参数

**输入/输出形状**:
- q: `(batch, num_q_heads, seq_len, head_dim)`
- k: `(batch, num_kv_heads, seq_len, head_dim)`
- 输出: (out_q, out_k)，与输入形状相同

```python
import numpy as np
from pyvqnet.tensor import QTensor, kfloat32
from pyvqnet.nn import RoPE

rope = RoPE(64, max_seq_len=128)
q = QTensor(np.random.randn(2, 8, 128, 64).astype(np.float32))
k = QTensor(np.random.randn(2, 4, 128, 64).astype(np.float32))
out_q, out_k = rope(q, k)
print(out_q.shape, out_k.shape)
```

**注意**: RoPE 通常需要 GPU（CUDA）后端运行。

---

## 激活函数

### ReLU

```python
pyvqnet.nn.ReLu(name="")
```

```python
from pyvqnet.nn import ReLu
from pyvqnet.tensor import QTensor

layer = ReLu()
y = layer(QTensor([-1, 2.0, -3, 4.0]))
print(y)
# [0., 2., 0., 4.]
```

### Sigmoid

```python
pyvqnet.nn.Sigmoid(name="")
```

### Tanh

```python
pyvqnet.nn.Tanh(name="")
```

### Softmax

```python
pyvqnet.nn.Softmax(dim=-1, name="")
```

### LeakyReLU

```python
pyvqnet.nn.LeakyReLu(alpha=0.01, name="")
```

### Gelu / GeLU

```python
pyvqnet.nn.Gelu(approximate="tanh", name="")
```

`GeLU` 是 `Gelu` 的别名。

### SiLU

```python
pyvqnet.nn.SiLU(name="")
```

Sigmoid Linear Unit（也称为 Swish）：`SiLU(x) = x * sigmoid(x)`。

```python
from pyvqnet.nn import SiLU
from pyvqnet.tensor import QTensor

layer = SiLU()
y = layer(QTensor([-1.0, 0.0, 1.0, 2.0]))
print(y)
```

### SwiGLU

```python
pyvqnet.nn.SwiGLU(name="")
```

SwiGLU 激活函数：`SwiGLU(gate, up) = SiLU(gate) * up`。常用于 LLM 的 FFN 层。接收两个输入张量（gate 和 up）。

```python
from pyvqnet.nn import SwiGLU
from pyvqnet.tensor import tensor

gate = tensor.randn([4, 128])
up = tensor.randn([4, 128])
layer = SwiGLU()
out = layer(gate, up)
print(out.shape)
```

### ELU

```python
pyvqnet.nn.ELU(alpha=1.0, name="")
```

指数线性单元（Exponential Linear Unit）。

### Softplus

```python
pyvqnet.nn.Softplus(name="")
```

`Softplus(x) = log(1 + exp(x))`，ReLU 的光滑近似。

### Softsign

```python
pyvqnet.nn.Softsign(name="")
```

`Softsign(x) = x / (1 + |x|)`。

### HardSigmoid

```python
pyvqnet.nn.HardSigmoid(name="")
```

分段线性近似 Sigmoid。

---

## 损失函数

**重要**: VQNet 损失函数参数顺序与 PyTorch 不同 - **第一个参数是目标值（标签），第二个参数是预测值**！

### MeanSquaredError

```python
pyvqnet.nn.MeanSquaredError(name="")
```

均方误差损失。

```python
from pyvqnet.tensor import QTensor
from pyvqnet import kfloat64
from pyvqnet.nn import MeanSquaredError

y = QTensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], requires_grad=False, dtype=kfloat64)
x = QTensor([[0.1, 0.05, 0.7, 0, 0.05, 0.1, 0, 0, 0, 0]], requires_grad=True, dtype=kfloat64)

loss_fn = MeanSquaredError()
result = loss_fn(y, x)  # 注意: (标签, 预测值)
print(result)
# [0.0115000]
```

### BinaryCrossEntropy

```python
pyvqnet.nn.BinaryCrossEntropy(name="")
```

二元交叉熵损失。

```python
from pyvqnet.tensor import QTensor
from pyvqnet.nn import BinaryCrossEntropy

x = QTensor([[0.3, 0.7, 0.2], [0.2, 0.3, 0.1]], requires_grad=True)
y = QTensor([[0.0, 1.0, 0], [0.0, 0, 1]], requires_grad=False)

loss_fn = BinaryCrossEntropy()
result = loss_fn(y, x)
print(result)
# [0.6364825]
```

### CategoricalCrossEntropy

```python
pyvqnet.nn.CategoricalCrossEntropy(name="")
```

分类交叉熵损失。

**注意**: 标签必须是 `kint64` 类型。

```python
from pyvqnet.tensor import QTensor
from pyvqnet import kfloat32, kint64
from pyvqnet.nn import CategoricalCrossEntropy

x = QTensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], requires_grad=True, dtype=kfloat32)
y = QTensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], requires_grad=False, dtype=kint64)

loss_fn = CategoricalCrossEntropy()
result = loss_fn(y, x)
print(result)
# [3.7852428]
```

### CrossEntropyLoss

```python
pyvqnet.nn.CrossEntropyLoss(name="")
```

计算 LogSoftmax 和 NLL_Loss 的组合损失。

### SoftmaxCrossEntropy

```python
pyvqnet.nn.SoftmaxCrossEntropy(name="")
```

带 Softmax 的交叉熵（数值更稳定）。

### NLL_Loss

```python
pyvqnet.nn.NLL_Loss(name="")
```

负对数似然损失（Negative Log Likelihood Loss），用于分类任务。输入需为 log-probabilities（通常由 LogSoftmax 产生）。

**注意**: 标签必须是 `kint64` 类型。

```python
from pyvqnet.tensor import QTensor
from pyvqnet import kint64
from pyvqnet.nn import NLL_Loss

x = QTensor([[0.9, 0.2, 0.1], [0.1, 0.8, 0.3]], requires_grad=True)
y = QTensor([0, 1], dtype=kint64)

loss_fn = NLL_Loss()
result = loss_fn(y, x)  # (标签, 预测值)
print(result)
```

---

## 优化器

### SGD

```python
pyvqnet.optim.SGD(params, lr=0.01, momentum=0, nesterov=False)
```

随机梯度下降。

```python
import numpy as np
from pyvqnet.optim import sgd
from pyvqnet.tensor import QTensor

w = np.arange(24).reshape(1, 2, 3, 4).astype(np.float64)
param = QTensor(w)
param.grad = QTensor(np.arange(24).reshape(1, 2, 3, 4).astype(np.float64))
params = [param]
opti = sgd.SGD(params)

for i in range(1, 3):
    opti._step()
    print(param)
```

### Adam

```python
pyvqnet.optim.Adam(params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0, amsgrad=False)
```

自适应矩估计优化器。

```python
import numpy as np
from pyvqnet.optim import adam
from pyvqnet.tensor import QTensor

w = np.arange(24).reshape(1, 2, 3, 4).astype(np.float64)
param = QTensor(w)
param.grad = QTensor(np.arange(24).reshape(1, 2, 3, 4).astype(np.float64))
params = [param]
opti = adam.Adam(params)

for i in range(1, 3):
    opti._step()
    print(param)
```

### AdamW

```python
pyvqnet.optim.AdamW(params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01, amsgrad=False)
```

带权重衰减的 Adam。

### Adagrad

```python
pyvqnet.optim.Adagrad(params, lr=0.01, epsilon=1e-8)
```

自适应梯度优化器。

### RMSProp

```python
pyvqnet.optim.RMSProp(params, lr=0.01, beta=0.99, epsilon=1e-8)
```

均方根传播优化器。

### Adadelta

```python
pyvqnet.optim.Adadelta(params, lr=0.01, beta=0.99, epsilon=1e-8)
```

### Adamax

```python
pyvqnet.optim.Adamax(params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
```

### Rotosolve

```python
pyvqnet.optim.Rotosolve(max_iter=50)
```

**量子参数优化器**。Rotosolve 算法用于优化量子测量期望值的线性组合（参考论文 [arXiv:1903.12166](https://arxiv.org/abs/1903.12166)）。

不需要梯度计算，通过参数移位（parameter shift）直接更新。

**参数**:
- `max_iter` - 最大迭代次数

**注意**: 
- 损失函数必须以 numpy 数组形式返回目标值（非 QTensor）
- 调用 `opt.minimize(params, costfunction)` 而非 `opt._step()`
- 适用于中小规模量子电路的参数优化

```python
from pyvqnet.optim.rotosolve import Rotosolve
from pyvqnet.tensor.tensor import QTensor
import numpy as np

def cost(params):
    # params is numpy array, return scalar
    return (params[0] - 0.5) ** 2 + (params[1] + 0.3) ** 2

t = QTensor([0.3, 0.25])
opt = Rotosolve(max_iter=10)
costs = opt.minimize(t, cost)
print(costs[-1])  # final cost
```

---

## 完整训练示例

```python
from pyvqnet.nn import Module, Linear, ReLU, Sequential
from pyvqnet.nn import MeanSquaredError
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
loss_fn = MeanSquaredError()

# 训练循环
x = randn([32, 4])
x.requires_grad = True
y = randn([32, 1])

pred = model(x)
loss = loss_fn(y, pred)  # 注意: (标签, 预测值)

optimizer.zero_grad()
loss.backward()
optimizer._step()

print(loss)
```

---

## GPU 训练

```python
from pyvqnet import DEV_GPU_0
from pyvqnet.nn import Module, Linear
from pyvqnet.optim import Adam
from pyvqnet.tensor import QTensor, randn

class Model(Module):
    def __init__(self):
        super().__init__()
        self.fc = Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

model = Model().toGPU(DEV_GPU_0)
optimizer = Adam(model.parameters(), lr=0.01)

# 数据也需要移动到 GPU
x = randn([32, 10], device=DEV_GPU_0)
x.requires_grad = True
y = randn([32, 5], device=DEV_GPU_0)

pred = model(x)
loss = ((pred - y) ** 2).mean()

optimizer.zero_grad()
loss.backward()
optimizer._step()
```

---

## Swin Transformer

Swin Transformer 层级式视觉 Transformer，使用滑动窗口注意力机制。

### SwinTransformer

```python
pyvqnet.nn.SwinTransformer(patch_size, embed_dim, depths, num_heads, window_size, mlp_ratio=4.0, dropout=0.0, attention_dropout=0.0, stochastic_depth_prob=0.1, num_classes=1000, norm_layer=None, block=None, downsample_layer=PatchMerging, feat_dim=16)
```

**参数**:
- `patch_size` - Patch 大小，如 `[4, 4]`
- `embed_dim` - Patch 嵌入维度
- `depths` - 各 Stage 的 Transformer Block 数量，如 `[2, 2, 18, 2]`
- `num_heads` - 各 Stage 注意力头数，如 `[4, 8, 16, 32]`
- `window_size` - 窗口大小，如 `[7, 7]`
- `mlp_ratio` - MLP 隐藏层维度与嵌入维度的比例
- `stochastic_depth_prob` - 随机深度概率
- `num_classes` - 分类数（`feat_dim` 控制最终特征维度）

```python
from pyvqnet.nn import SwinTransformer
from pyvqnet.tensor import tensor

model = SwinTransformer(
    patch_size=[4, 4],
    embed_dim=128,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    window_size=[7, 7],
    stochastic_depth_prob=0.5,
)
x = tensor.randn([1, 3, 224, 224])
y = model(x)
print(y.shape)
```

### swin_b

```python
pyvqnet.nn.swin_b(weights=None, **kwargs)
```

Swin-Base 预配置模型（等同 SwinTransformer-B）。参数:
- patch_size=[4, 4], embed_dim=128, depths=[2, 2, 18, 2]
- num_heads=[4, 8, 16, 32], window_size=[7, 7]
- stochastic_depth_prob=0.5

```python
from pyvqnet.nn import swin_b
model = swin_b()
```

---

## LLM Token 采样函数

LLM 推理中的 token 采样函数，位于 `pyvqnet.nn.functional`。**需要 GPU（CUDA）后端**。

### top_k_top_p_sampling_from_logits

```python
pyvqnet.nn.functional.top_k_top_p_sampling_from_logits(logits, temperature=1.0, top_k=0, top_p=1.0, deterministic=True) -> (sampled_tokens, valid)
```

端到端 token 采样：top_k 过滤 → temperature 缩放 → softmax → top_p 拒绝采样。

**参数**:
- `logits` - 形状 `(batch_size, vocab_size)`，float32/float64/bf16
- `temperature` - 温度参数（标量或 `(batch_size,)` 张量）
- `top_k` - Top-K 阈值，0 表示不限制
- `top_p` - Top-P（nucleus）阈值，1.0 表示不限制
- `deterministic` - 是否确定性扫描
- 返回: `(sampled_tokens, valid)` - tokens 为 int64 形状 `(batch_size,)`，valid 为 bool

```python
import pyvqnet
from pyvqnet.tensor import QTensor
from pyvqnet.nn.functional import top_k_top_p_sampling_from_logits

pyvqnet.backends.set_backend("pyvqnet-ad")
logits = QTensor([[0.1, 0.2, 0.5, 0.1, 0.1]], device="gpu:0")
tokens, valid = top_k_top_p_sampling_from_logits(logits, temperature=0.8, top_k=3, top_p=0.9)
print(tokens.to_numpy())  # e.g. [2]
```

### top_k_top_p_sampling_from_probs

```python
pyvqnet.nn.functional.top_k_top_p_sampling_from_probs(probs, top_k=0, top_p=1.0, deterministic=True) -> (sampled_tokens, valid)
```

从概率分布中进行联合 Top-K + Top-P 采样（不排序整个词汇表，使用 pivot-convergence 拒绝采样）。

### top_p_sampling_from_probs

```python
pyvqnet.nn.functional.top_p_sampling_from_probs(probs, top_p=1.0, deterministic=True) -> (sampled_tokens, valid)
```

仅 Top-P（nucleus）采样。

### top_k_sampling_from_probs

```python
pyvqnet.nn.functional.top_k_sampling_from_probs(probs, top_k=0, deterministic=True) -> (sampled_tokens, valid)
```

仅 Top-K 采样。

### min_p_sampling_from_probs

```python
pyvqnet.nn.functional.min_p_sampling_from_probs(probs, min_p=0.0, deterministic=True) -> (sampled_tokens, valid)
```

Min-P 采样：仅保留 `prob >= max_prob * min_p` 的 token，然后从中采样。

```python
import pyvqnet
from pyvqnet.tensor import QTensor
from pyvqnet.nn.functional import min_p_sampling_from_probs

pyvqnet.backends.set_backend("pyvqnet-ad")
probs = QTensor([[0.1, 0.3, 0.5, 0.1]], device="gpu:0")
tokens, valid = min_p_sampling_from_probs(probs, min_p=0.1)
print(tokens.to_numpy())  # e.g. [2]
```

**注意**: LLM 采样函数为纯推理操作（无 autograd），需要在 GPU 上运行。

---

## Interpolate 插值模块

```python
pyvqnet.nn.Interpolate(size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None, name="")
```

上采样/下采样模块，支持 `"nearest"`、`"bilinear"`、`"bicubic"` 模式。

```python
from pyvqnet.nn import Interpolate
from pyvqnet.tensor import tensor
import pyvqnet

model = Interpolate(size=3, mode="bilinear")
input_vqnet = tensor.randu((1, 1, 6, 6), dtype=pyvqnet.kfloat32, requires_grad=True)
output_vqnet = model(input_vqnet)
print(output_vqnet.shape)
```

---

## Pixel Shuffle / Unshuffle

### Pixel_Shuffle

```python
pyvqnet.nn.Pixel_Shuffle(upscale_factors, name="")
```

将形状 `(*, C × r², H, W)` 重排为 `(*, C, H × r, W × r)`，用于超分辨率上采样。

### Pixel_Unshuffle

```python
pyvqnet.nn.Pixel_Unshuffle(downscale_factors, name="")
```

Pixel_Shuffle 的逆操作，将 `(*, C, H × r, W × r)` 重排为 `(*, C × r², H, W)`。

```python
from pyvqnet.nn import Pixel_Shuffle
from pyvqnet.tensor import tensor

ps = Pixel_Shuffle(3)
inx = tensor.ones([5, 2, 3, 18, 4, 4])
y = ps(inx)
print(y.shape)
```

---

## Identity 恒等层

```python
pyvqnet.nn.Identity(name="")
```

占位恒等运算符，返回输入本身。用于模型结构中的占位或条件分支。

```python
from pyvqnet.nn import Identity
layer = Identity()
x = QTensor([1.0, 2.0, 3.0])
y = layer(x)  # 返回 x 本身
```

---

## 量子经典混合层

### QLinear

```python
pyvqnet.qnn.qlinear.QLinear(input_channels, output_channels, machine="CPU")
```

量子全连接层。

```python
from pyvqnet.tensor import QTensor
from pyvqnet.qnn.qlinear import QLinear

params = [[0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
          [1.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
          [1.37454012, 1.95071431, 0.73199394, 1.59865848, 0.15601864, 0.15599452],
          [1.37454012, 1.95071431, 1.73199394, 1.59865848, 0.15601864, 0.15599452]]

m = QLinear(6, 2)
input = QTensor(params, requires_grad=True)
output = m(input)
output.backward()
print(output)
```

### QConv

```python
pyvqnet.qnn.qcnn.qconv.QConv(input_channels, output_channels, quantum_number, stride=(1, 1), padding=(0, 0), kernel_initializer=normal, machine="CPU", dtype=None, name="")
```

量子卷积层。

```python
from pyvqnet.tensor import tensor
from pyvqnet.qnn.qcnn.qconv import QConv

x = tensor.ones([1, 3, 4, 4])
layer = QConv(input_channels=3, output_channels=2, quantum_number=4, stride=(2, 2))
y = layer(x)
print(y)
```

---

## 常见问题

1. **损失函数参数顺序**: VQNet 损失函数 `loss_fn(y_true, y_pred)` - 第一个是标签，第二个是预测值
2. **dtype 错误**: Embedding 和 CrossEntropy 的标签需要 `kint64`
3. **ModuleList vs list**: 子模块必须用 `ModuleList`，不能用 Python `list`
4. **GPU 训练**: 模型和数据都要移动到 GPU
5. **zero_grad**: 训练前要调用 `optimizer.zero_grad()` 清零梯度
6. **优化器更新**: `optimizer._step()` 与 `optimizer.step()` 均可用且效果等价（官方示例多写 `_step()`，官方类型标注 `.pyi` 声明的是 `step()`）
7. **Rotosolve 用法**: 调用 `opt.minimize(params, costfunction)` 而非 `_step()`，损失函数返回 numpy 数组
8. **RoPE 硬件需求**: RoPE 通常需要 GPU（CUDA）后端运行
9. **LLM 采样硬件需求**: `top_k_top_p_sampling_from_logits` 等采样函数需要 GPU（CUDA）后端，且为纯推理（无 autograd）
10. **SwiGLU 双输入**: SwiGLU 接收两个参数 `(gate, up)`，不是单输入激活函数
11. **RMSNorm vs LayerNorm**: RMSNorm 无均值中心化，无 bias 参数，计算更轻量

---

**Version**: VQNet 2.18.1
