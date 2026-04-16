# Classical Neural Network API Reference

> 来源: VQNET2.0-tutorial/source/rst/nn.rst
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

---

## Dropout

```python
pyvqnet.nn.Dropout(dropout_rate=0.5)
```

随机丢弃神经元。

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
pyvqnet.nn.Softmax(axis=-1, name="")
```

### LeakyReLU

```python
pyvqnet.nn.LeakyReLu(alpha=0.01, name="")
```

### Gelu

```python
pyvqnet.nn.Gelu(approximate="tanh", name="")
```

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
6. **_step**: VQNet 用 `optimizer._step()` 而不是 `step()`

---

**Version**: VQNet 2.0
**Source**: VQNET2.0-tutorial/source/rst/nn.rst