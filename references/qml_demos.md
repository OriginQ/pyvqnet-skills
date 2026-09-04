# QML Demo 示例代码

> **重要**: 所有示例代码均来自官方文档。

---

## QVC 量子变分分类器

论文: *Circuit-centric quantum classifiers* https://arxiv.org/pdf/1804.00633.pdf

### 任务
判断二进制数是奇数还是偶数。

### 量子电路结构

编码: 将二进制输入编码到量子比特
```
x = 0101 → |ψ⟩ = |0101⟩
```

变分层: RZ + RY + RZ + CNOT

### 完整代码

```python
import pyqpanda3.core as pq
from pyvqnet.nn.module import Module
from pyvqnet.optim.sgd import SGD
from pyvqnet.nn.loss import CategoricalCrossEntropy
from pyvqnet.tensor.tensor import QTensor
from pyvqnet.data import data_generator as dataloader
from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
from pyvqnet.qnn.pq3.measure import probs_measure
import numpy as np

qnum = 4

def qvc_circuits(input, weights):
    qlist = range(qnum)
    machine = pq.CPUQVM()

    def get_cnot(nqubits):
        cir = pq.QCircuit()
        for i in range(len(nqubits)-1):
            cir << pq.CNOT(nqubits[i], nqubits[i+1])
        cir << pq.CNOT(nqubits[len(nqubits)-1], nqubits[0])
        return cir

    def Rot(weights_j, qubits):
        circult = pq.QCircuit()
        circult << pq.RZ(qubits, weights_j[0])
        circult << pq.RY(qubits, weights_j[1])
        circult << pq.RZ(qubits, weights_j[2])
        return circult

    def basisstate():
        circult = pq.QCircuit()
        for i in range(len(qlist)):
            if input[i] == 1:
                circult << pq.X(qlist[i])
        return circult

    circult = pq.QCircuit()
    circult << basisstate()

    weights = weights.reshape([2, 4, 3])
    for i in range(weights.shape[0]):
        weights_i = weights[i, :, :]
        for j in range(len(qlist)):
            weights_j = weights_i[j]
            circult << Rot(weights_j, qlist[j])
        circult << get_cnot(qlist)

    prog = pq.QProg()
    prog << circult
    prob = probs_measure(machine, prog, qlist[0])
    return prob

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.qvc = QuantumLayer(qvc_circuits, 24)

    def forward(self, x):
        return self.qvc(x)

# 训练数据
qvc_train_data = [0,1,0,0,1, 0,1,0,1,0, 0,1,1,0,0, 0,1,1,1,1,
                  1,0,0,0,1, 1,0,0,1,0, 1,0,1,0,0, 1,0,1,1,1,
                  1,1,0,0,0, 1,1,0,1,1, 1,1,1,0,1, 1,1,1,1,0]

qvc_test_data = [0,0,0,0,0, 0,0,0,1,1, 0,0,1,0,1, 0,0,1,1,0]

def get_data(dataset_str):
    if dataset_str == "train":
        datasets = np.array(qvc_train_data)
    else:
        datasets = np.array(qvc_test_data)
    datasets = datasets.reshape([-1, 5])
    data = datasets[:, :-1]
    label = datasets[:, -1].astype(int)
    label = np.eye(2)[label].reshape(-1, 2)
    return data, label

def get_accuary(result, label):
    result, label = np.array(result.data), np.array(label.data)
    score = np.sum(np.argmax(result, axis=1) == np.argmax(label, 1))
    return score

model = Model()
optimizer = SGD(model.parameters(), lr=0.1)
batch_size = 3
epoch = 20
loss = CategoricalCrossEntropy()

model.train()
datas, labels = get_data("train")

for i in range(epoch):
    count = 0
    sum_loss = 0
    accuary = 0
    for data, label in dataloader(datas, labels, batch_size, False):
        optimizer.zero_grad()
        data, label = QTensor(data), QTensor(label)
        result = model(data)
        loss_b = loss(label, result)
        loss_b.backward()
        optimizer._step()
        sum_loss += loss_b.item()
        count += batch_size
        accuary += get_accuary(result, label)
    print(f"epoch:{i}, loss:{sum_loss/count}, accuracy:{accuary/count}")

model.eval()
test_data, test_label = get_data("test")
test_accuary = 0
for testd, testl in dataloader(test_data, test_label, 1):
    testd = QTensor(testd)
    test_result = model(testd)
    test_accuary += get_accuary(test_result, testl)
print(f"test accuracy:{test_accuary/len(test_data)}")
```

---

## QDRL 数据重上传模型

论文: *Data re-uploading for a universal quantum classifier* https://arxiv.org/abs/1907.02085

### 原理
数据在幺正变换之前需要进行重新上传操作，类似于经典神经网络中每个神经元接受上层所有信息。

### 完整代码

```python
import numpy as np
from pyvqnet.nn.linear import Linear
from pyvqnet.qnn.qdrl.vqnet_model import vmodel
from pyvqnet.optim import sgd
from pyvqnet.nn.loss import CategoricalCrossEntropy
from pyvqnet.tensor.tensor import QTensor
from pyvqnet.nn.module import Module
from pyvqnet.data import data_generator as get_minibatch_data

np.random.seed(42)
num_layers = 3
params = np.random.uniform(size=(num_layers, 3))

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.pqc = vmodel(params.shape)
        self.fc2 = Linear(2, 2)

    def forward(self, x):
        x = self.pqc(x)
        x = self.fc2(x)
        return x

model = Model()
optimizer = sgd.SGD(model.parameters(), lr=0.1)
loss_func = CategoricalCrossEntropy()
batch_size = 25
epoch = 20

# 生成数据
def circle(samples):
    data_x, data_y = [], []
    for i in range(samples):
        x = 2 * np.random.rand(2) - 1
        y = [0, 1]
        if np.linalg.norm(x) < 1:
            y = [1, 0]
        data_x.append(x)
        data_y.append(y)
    return np.array(data_x), np.array(data_y)

x_train, y_train = circle(500)
x_train = np.hstack((x_train, np.zeros((x_train.shape[0], 1))))

model.train()
for i in range(epoch):
    for data, label in get_minibatch_data(x_train, y_train, batch_size, True):
        optimizer.zero_grad()
        data = QTensor(data)
        label = QTensor(label)
        output = model(data)
        loss_b = loss_func(label, output)
        loss_b.backward()
        optimizer._step()
```

---

## VSQL 变分量子线路

基于 VQC 模块的变分量子线路分类器。

---

## Quanvolution 量子卷积

使用量子电路替代卷积核进行图像处理。

### 关键 API

```python
from pyvqnet.qnn.qcnn.qconv import QConv

layer = QConv(
    input_channels=3,
    output_channels=2,
    quantum_number=4,  # 每个卷积核使用的量子比特数
    stride=(2, 2)
)
```

---

## 混合 CNN + QNN 模型

```python
from pyvqnet.nn import Module, Conv2D, Linear, ReLU, Sequential
from pyvqnet.qnn.vqc import QMachine, RZ, Probability
from pyvqnet.qnn.vqc import VQC_HardwareEfficientAnsatz

class HybridModel(Module):
    def __init__(self):
        super().__init__()
        # 经典 CNN 部分
        self.cnn = Sequential(
            Conv2D(1, 16, 3),
            ReLU(),
            Conv2D(16, 32, 3)
        )
        # 量子部分
        self.device = QMachine(4)
        self.ansatz = VQC_HardwareEfficientAnsatz(
            4, ["rx", "RY", "rz"],
            entangle_gate="cnot",
            depth=2
        )
        self.measure = Probability(wires=[0, 1])

    def forward(self, x):
        # 经典前处理
        x = self.cnn(x)
        x = x.reshape([-1, 32])

        # 量子计算
        self.device.reset_states(x.shape[0])
        self.ansatz(q_machine=self.device)
        return self.measure(q_machine=self.device)
```

---

## QKMeans 量子 K-Means 聚类

使用量子 SWAP 测试（`QKmeansCircuits`）计算数据点与质心之间的距离，实现量子 K-Means 无监督聚类算法。

### 关键 API

```python
from pyvqnet.qnn.qkmeans import QKmeans
from pyvqnet.qnn.qkmeans.circuit import QKmeansCircuits
```

### 完整代码

```python
from pyvqnet.qnn.qkmeans import QKmeans

qkmeans = QKmeans(k=3, epoch=5, num_qubits=3)
qkmeans.run(n=100, std=2)
```

量子 K-Means 核心在于使用量子 SWAP 测试电路度量距离：将数据点和质心编码为量子态，通过受控 SWAP 门和 H 门测量得到两个态的重叠度（保真度），从而计算相似度。电路使用 3 个量子比特，通过 H 门初始化、U3 旋转编码、受控 SWAP 交换和测量实现。

---

## Quantum Expressibility 量子电路表达能力

评估参数化量子电路（PQC）对 Hilbert 空间的探索能力（Expressibility）。计算电路输出态与 Haar 随机分布之间的 KL 散度，值越小表示表达能力越强。

### 关键 API

```python
from pyvqnet.qnn.quantum_expressibility import fidelity_of_cir, fidelity_harr_sample
from pyvqnet.qnn.ansatz import HardwareEfficientAnsatz
```

### 完整代码

```python
import numpy as np
from scipy.stats import entropy
from pyvqnet.qnn.quantum_expressibility import fidelity_of_cir, fidelity_harr_sample
from pyvqnet.qnn.ansatz import HardwareEfficientAnsatz

num_qubit = 4
num_sample = 2000

# Haar 随机分布的保真度采样（作为理论基准）
flist, p_haar, theory_haar = fidelity_harr_sample(num_qubit, num_sample)

# 计算不同深度 HardwareEfficientAnsatz 的表达能力
for depth in range(1, 6):
    f_list, p_cel = fidelity_of_cir(
        HardwareEfficientAnsatz, num_qubit, depth, num_sample
    )
    # KL 散度越小 → 表达能力越强（越接近 Haar 分布）
    expr = entropy(p_cel, theory_haar)
    print(f"Depth {depth}: Expressibility = {expr:.4f}")
```

---

## TTOLayer 张量训练算子层

基于张量训练（Tensor Train, TT）分解的高效神经网络层，将高维权重矩阵分解为多个低秩核心张量，显著减少参数量和计算复杂度。

### 关键 API

```python
from pyvqnet.qnn.ttolayer import TTOLayer
```

### 构造函数参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `inp_modes` | list[int] | 输入张量各维度大小 |
| `out_modes` | list[int] | 输出张量各维度大小 |
| `mat_ranks` | list[int] | TT 分解的秩（首尾必须为 1） |
| `biases_initializer` | callable | 偏置初始化函数（默认 tensor.zeros） |

### 完整代码

```python
import numpy as np
from pyvqnet.tensor import QTensor
from pyvqnet.qnn.ttolayer import TTOLayer
from pyvqnet.dtype import kfloat32

inp_modes = [4, 5]
out_modes = [4, 5]
mat_ranks = [1, 3, 1]  # 首尾必须为 1

tto_layer = TTOLayer(inp_modes, out_modes, mat_ranks)

batch_size = 2
seq_len = 4
embed_size = 5
inp = QTensor(np.random.randn(batch_size, seq_len, embed_size), dtype=kfloat32)

output = tto_layer(inp)
print("Input shape:", inp.shape)
print("Output shape:", output.shape)
```

TTOLayer 将输入 `[batch, len, embed]` 重塑为 TT 格式，通过 `mat_cores`（`ParameterList`）逐模式矩阵乘法变换后恢复原形状，并可选加偏置。

---

## QDRL_VQC 基于 VQC 的数据重上传模型

与 `pyvqnet.qnn.qdrl.vqnet_model.vmodel`（基于 QuantumLayer）不同，QDRL_VQC 使用 VQC 自动微分模块（`QModule`/`QMachine`），在 forward 中通过 `ry()`、`rz()` 等门函数构建电路，并需显式调用 `reset_states()`。

### 关键 API

```python
from pyvqnet.qnn.qdrl_vqc.qdrl_vqc import QDRL
```

### 完整代码

```python
import numpy as np
from pyvqnet.nn.module import Module
from pyvqnet.nn.linear import Linear
from pyvqnet.nn.loss import CategoricalCrossEntropy
from pyvqnet.optim import sgd
from pyvqnet.tensor.tensor import QTensor
from pyvqnet.data import data_generator as get_minibatch_data
from pyvqnet.qnn.qdrl_vqc.qdrl_vqc import QDRL

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.qdrl = QDRL(nq=1)  # 1 个量子比特
        self.fc2 = Linear(2, 2)

    def forward(self, x):
        x = self.qdrl(x)  # QDRL 内部已调用 reset_states
        x = self.fc2(x)
        return x

# 生成圆形分类数据
def circle(samples):
    data_x, data_y = [], []
    for _ in range(samples):
        x = 2 * np.random.rand(2) - 1
        y = [0, 1]
        if np.linalg.norm(x) < 1:
            y = [1, 0]
        data_x.append(x)
        data_y.append(y)
    return np.array(data_x), np.array(data_y)

model = Model()
optimizer = sgd.SGD(model.parameters(), lr=0.1)
loss_func = CategoricalCrossEntropy()

x_train, y_train = circle(500)
x_train = np.hstack((x_train, np.zeros((x_train.shape[0], 1))))

model.train()
for i in range(20):
    for data, label in get_minibatch_data(x_train, y_train, 25, True):
        optimizer.zero_grad()
        data = QTensor(data)
        label = QTensor(label)
        output = model(data)
        loss_b = loss_func(label, output)
        loss_b.backward()
        optimizer._step()
```

### QDRL vs QDRL_VQC 对比

| 特性 | QDRL (pyvqnet.qnn.qdrl) | QDRL_VQC (pyvqnet.qnn.qdrl_vqc) |
|------|------------------------|-------------------------------|
| 基类 | `vmodel`（基于 QuantumLayer） | `QModule`（基于 VQC 自动微分） |
| 导入路径 | `from pyvqnet.qnn.qdrl.vqnet_model import vmodel` | `from pyvqnet.qnn.qdrl_vqc.qdrl_vqc import QDRL` |
| 量子门构建 | 在电路函数中通过 pyqpanda3 构建 | `ry()`, `rz()` 等 VQC 门函数 |
| reset_states | 不需要 | forward 中必须显式调用 |
| 参数定义 | 通过 `params` shape 传入 | 通过 `Parameter` 对象 |

---

## 常见 QML 模型类型

| 模型 | 应用场景 |
|------|----------|
| QVC | 二分类/多分类 |
| VSQL | 序列分类 |
| Quanvolution | 图像处理 |
| QAE | 数据压缩/生成 |
| QGAN | 生成任务 |
| QKMeans | 无监督聚类 |
| Quantum Expressibility | 电路表达能力分析 |
| TTOLayer | 张量分解高效神经网络层 |
| QDRL_VQC | 基于 VQC 的数据重上传分类 |
| Hybrid CNN+QNN | 混合量子经典模型 |

---

**Version**: VQNet 2.18.1
