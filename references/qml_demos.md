# QML Demo 示例代码

> 来源: VQNET2.0-tutorial/source/rst/qml_demo.rst + vqc_demo.rst
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
from pyvqnet.qnn.vqc.qcircuit import VQC_HardwareEfficientAnsatz

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

## 常见 QML 模型类型

| 模型 | 应用场景 |
|------|----------|
| QVC | 二分类/多分类 |
| VSQL | 序列分类 |
| Quanvolution | 图像处理 |
| QAE | 数据压缩/生成 |
| QGAN | 生成任务 |
| Hybrid CNN+QNN | 混合量子经典模型 |

---

**Version**: VQNet 2.0
**Source**: VQNET2.0-tutorial/source/rst/qml_demo.rst + vqc_demo.rst