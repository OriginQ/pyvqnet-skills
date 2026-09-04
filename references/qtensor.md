# QTensor API Reference

> **重要**: 所有示例代码均来自官方文档，可直接运行。

---

## QTensor 类定义

```python
pyvqnet.tensor.tensor.QTensor(
    data,
    requires_grad=False,
    nodes=None,
    device=pyvqnet.DEV_CPU,
    dtype=None,
    name=""
)
```

**参数**:
- `data` - 输入数据，可以是 numpy 数组或列表
- `requires_grad` - 是否跟踪梯度，默认 False
- `device` - 存储设备，默认 `pyvqnet.DEV_CPU`
- `dtype` - 数据类型，默认 `kfloat32`

**示例**:
```python
from pyvqnet.tensor import QTensor
from pyvqnet.dtype import *
import numpy as np

t1 = QTensor(np.ones([2,3]))
t2 = QTensor([2,3,4j,5])
t3 = QTensor([[[2,3,4,5],[2,3,4,5]]], dtype=kbool)
print(t1)
# [[1. 1. 1.]
#  [1. 1. 1.]]
print(t2)
# [2.+0.j 3.+0.j 0.+4.j 5.+0.j]
print(t3)
# [[[ True  True  True  True]
#   [ True  True  True  True]]]
```

---

## QTensor 属性

### ndim
返回张量的维度个数。

```python
from pyvqnet.tensor import QTensor

a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
print(a.ndim)
# 1
```

### shape
返回张量的维度列表。

```python
from pyvqnet.tensor import QTensor

a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
print(a.shape)
# [4]
```

### size
返回张量的元素个数。

```python
from pyvqnet.tensor import QTensor

a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
print(a.size)
# 4
```

### dtype
返回张量的数据类型。

**支持的数据类型**:
| dtype | 描述 |
|-------|------|
| `pyvqnet.kbool` | 布尔变量 |
| `pyvqnet.kuint8` | 8比特整数 (无符号) |
| `pyvqnet.kint8` | 8比特整数 (有符号) |
| `pyvqnet.kint16` | 16比特整数 |
| `pyvqnet.kint32` | 32比特整数 |
| `pyvqnet.kint64` | 64比特整数 |
| `pyvqnet.kfloat32` | 32比特浮点数 (默认) |
| `pyvqnet.kfloat64` | 64比特浮点数 |
| `pyvqnet.kcomplex64` | 64比特复数 |
| `pyvqnet.kcomplex128` | 128比特复数 |
| `pyvqnet.kbfloat16` | 16比特脑浮点格式 |

```python
from pyvqnet.tensor import QTensor

a = QTensor([2, 3, 4, 5])
print(a.dtype)
# 4 (kfloat32)
```

### is_contiguous
判断是否是 contiguous 的多维数组。

```python
from pyvqnet.tensor import QTensor

a = QTensor([[2, 3, 4, 5],[2, 3, 4, 5]])
print(a.is_contiguous)
# True
c = a.permute((1,0))
print(c.is_contiguous)
# False
```

### requires_grad
是否跟踪梯度。

### grad
梯度值，在 `backward()` 后可用。

---

## QTensor 方法

### zero_grad()
将梯度设置为零。

```python
from pyvqnet.tensor import QTensor

t3 = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
t3.zero_grad()
print(t3.grad)
# [0., 0., 0., 0.]
```

### backward(grad=None)
反向传播计算梯度。

```python
from pyvqnet.tensor import QTensor

target = QTensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0.2]], requires_grad=True)
y = 2*target + 3
y.backward()
print(target.grad)
# [[2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]]
```

### to_numpy()
转换为 numpy 数组。

```python
from pyvqnet.tensor import tensor, QTensor

t3 = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
t4 = t3.to_numpy()
print(t4)
# [2. 3. 4. 5.]
```

### item()
返回单个元素值。

```python
from pyvqnet.tensor import tensor

t = tensor.ones([1])
print(t.item())
# 1.0
```

### reshape(new_shape)
改变形状。

```python
from pyvqnet.tensor import tensor, QTensor
import numpy as np

R, C = 3, 4
a = np.arange(R * C).reshape(R, C).astype(np.float32)
t = QTensor(a)
reshape_t = t.reshape([C, R])
print(reshape_t)
# [
# [0., 1., 2.],
# [3., 4., 5.],
# [6., 7., 8.],
# [9., 10., 11.]
# ]
```

### transpose(new_dims=None)
反转张量的轴。

```python
from pyvqnet.tensor import tensor, QTensor
import numpy as np

R, C = 3, 4
a = np.arange(R * C).reshape([2, 2, 3]).astype(np.float32)
t = QTensor(a)
rlt = t.transpose([2,0,1])
print(rlt)
# [
# [[0., 3.],
#  [6., 9.]],
# [[1., 4.],
#  [7., 10.]],
# [[2., 5.],
#  [8., 11.]]
# ]
```

### GPU(device=DEV_GPU_0) / toGPU()
移动到 GPU。

```python
from pyvqnet.tensor import QTensor

a = QTensor([2])
b = a.GPU()
print(b.device)
# 1000
```

### CPU() / toCPU()
移动到 CPU。

```python
from pyvqnet.tensor import QTensor

a = QTensor([2])
b = a.CPU()
print(b.device)
# 0
```

### astype(dtype)
转换数据类型。

```python
from pyvqnet.tensor import QTensor
from pyvqnet import kcomplex128

a = QTensor([2])
a = a.astype(kcomplex128)
print(a)
# [2.+0.j]
```

### argmax/argmin
返回最大/最小值的索引。

```python
from pyvqnet.tensor import tensor, QTensor

a = QTensor([[1.3398, 0.2663, -0.2686, 0.2450],
            [-0.7401, -0.8805, -0.3402, -1.1936],
            [0.4907, -1.3948, -1.0691, -0.3132],
            [-1.6092, 0.5419, -0.2993, 0.3195]])
flag = a.argmax()
print(flag)
# [0.]

flag_0 = a.argmax([0], True)
print(flag_0)
# [[0., 3., 0., 3.]]

flag_1 = a.argmax([1], True)
print(flag_1)
# [[0.], [2.], [0.], [1.]]
```

---

## 创建函数

### ones(shape, device=DEV_CPU, dtype=None)
创建全 1 张量。

```python
from pyvqnet.tensor import tensor

x = tensor.ones([2, 3])
print(x)
# [[1., 1., 1.],
#  [1., 1., 1.]]
```

### zeros(shape, device=DEV_CPU, dtype=None)
创建全 0 张量。

```python
from pyvqnet.tensor import tensor, QTensor

t = tensor.zeros([2, 3, 4])
print(t)
# [[[0., 0., 0., 0.],
#   [0., 0., 0., 0.],
#   [0., 0., 0., 0.]],
#  [[0., 0., 0., 0.],
#   [0., 0., 0., 0.],
#   [0., 0., 0., 0.]]]
```

### arange(start, end, step=1, device=DEV_CPU, dtype=None, requires_grad=False)
创建等间隔序列。

```python
from pyvqnet.tensor import tensor, QTensor

t = tensor.arange(2, 30, 4)
print(t)
# [ 2.,  6., 10., 14., 18., 22., 26.]
```

### linspace(start, end, num, device=DEV_CPU, dtype=None, requires_grad=False)
创建线性间隔序列。

```python
from pyvqnet.tensor import tensor, QTensor

start, stop, num = -2.5, 10, 10
t = tensor.linspace(start, stop, num)
print(t)
# [-2.5000000, -1.1111112, 0.2777777, 1.6666665, 3.0555553,
#  4.4444442, 5.8333330, 7.2222219, 8.6111107, 10.]
```

### randn(shape, mean=0.0, std=1.0, device=DEV_CPU, dtype=None, requires_grad=False)
创建正态分布随机张量。

```python
from pyvqnet.tensor import tensor, QTensor

shape = [2, 3]
t = tensor.randn(shape)
print(t)
# [[-0.9529880, -0.4947567, -0.6399882],
#  [-0.6987777, -0.0089036, -0.5084590]]
```

### randu(shape, min=0.0, max=1.0, device=DEV_CPU, dtype=None, requires_grad=False)
创建均匀分布随机张量。

```python
from pyvqnet.tensor import tensor, QTensor

shape = [2, 3]
t = tensor.randu(shape)
print(t)
# [[0.0885886, 0.9570093, 0.8304565],
#  [0.6055251, 0.8721224, 0.1927866]]
```

### full(shape, value, device=DEV_CPU, dtype=None)
创建填充特定值的张量。

```python
from pyvqnet.tensor import tensor, QTensor

shape = [2, 3]
value = 42
t = tensor.full(shape, value)
print(t)
# [[42., 42., 42.],
#  [42., 42., 42.]]
```

### eye(size, offset=0, device=DEV_CPU, dtype=None)
创建单位矩阵。

```python
from pyvqnet.tensor import tensor, QTensor

size = 3
t = tensor.eye(size)
print(t)
# [[1., 0., 0.],
#  [0., 1., 0.],
#  [0., 0., 1.]]
```

### ones_like / zeros_like / full_like
创建与输入形状相同的张量。

```python
from pyvqnet.tensor import tensor, QTensor

t = QTensor([1, 2, 3])
x = tensor.ones_like(t)
print(x)
# [1., 1., 1.]
```

---

## 数学运算

### add / sub / mul / divide
元素级加减乘除。

```python
from pyvqnet.tensor import tensor, QTensor

t1 = QTensor([1, 2, 3])
t2 = QTensor([4, 5, 6])
print(tensor.add(t1, t2))   # [5., 7., 9.]
print(tensor.sub(t1, t2))   # [-3., -3., -3.]
print(tensor.mul(t1, t2))   # [4., 10., 18.]
print(tensor.divide(t1, t2))# [0.25, 0.4, 0.5]
```

### sums / mean / std / var
聚合统计。

```python
from pyvqnet.tensor import tensor, QTensor

t = QTensor([[1, 2, 3], [4, 5, 6]])
print(tensor.sums(t))       # [21.]
print(tensor.mean(t, 1))    # [2., 5.]
```

### matmul
矩阵乘法。

```python
from pyvqnet.tensor import tensor

t1 = tensor.ones([2,3])
t1.requires_grad = True
t2 = tensor.ones([3,4])
t2.requires_grad = True
t3 = tensor.matmul(t1, t2)
t3.backward(tensor.ones_like(t3))
print(t1.grad)
# [[4., 4., 4.],
#  [4., 4., 4.]]
print(t2.grad)
# [[2., 2., 2., 2.],
#  [2., 2., 2., 2.],
#  [2., 2., 2., 2.]]
```

### floor / ceil / round
取整函数。

```python
from pyvqnet.tensor import tensor

t = tensor.arange(-2.0, 2.0, 0.25)
print(tensor.floor(t))
# [-2., -2., -2., -2., -1., -1., -1., -1., 0., 0., 0., 0., 1., 1., 1., 1.]
print(tensor.ceil(t))
# [-2., -1., -1., -1., -1., -0., -0., -0., 0., 1., 1., 1., 1., 2., 2., 2.]
```

---

## 形状操作

### permute
维度重排。

### flatten
展平。

### concatenate / stack
拼接/堆叠。

### split
分割。

---

## 切片索引

```python
from pyvqnet.tensor import tensor, QTensor

aaa = tensor.arange(1, 61)
aaa = aaa.reshape([4, 5, 3])
print(aaa[0:2, 3, :2])
# [[10., 11.], [25., 26.]]
print(aaa[3, 4, 1])
# [59.]
print(aaa[:, 2, :])
# [[7., 8., 9.],
#  [22., 23., 24.],
#  [37., 38., 39.],
#  [52., 53., 54.]]

# 布尔索引
a = tensor.ones([2, 2])
b = QTensor([[1, 1], [0, 1]])
b = b > 0
c = a[b]
print(c)
# [1., 1., 1.]
```

---

## 常见问题

1. **dtype 错误**: Embedding 层输入需要 `kint64`
2. **梯度为 None**: 确保 `requires_grad=True`
3. **GPU/CPU 混用**: 同一计算图中的张量必须在同一设备

---

**Version**: VQNet 2.0
