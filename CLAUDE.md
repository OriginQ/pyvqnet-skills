# CLAUDE.md - VQNet 2.0 API Skill

## Project Overview

**Purpose**: AI skill package for helping users develop quantum machine learning models using VQNet 2.0 (PyVQNet) with pyqpanda3. Provides complete API reference extracted from official documentation.

**Primary Framework**: VQNet 2.0 (PyVQNet) with pyqpanda3 for quantum computing.

**Target Users**: AI assistants (Claude/Cline) helping users with quantum machine learning using VQNet 2.0.

**License**: Apache 2.0

---

## Critical Rule: Always Read Reference Files

**IMPORTANT**: This Skill enforces a strict rule to prevent API hallucination:

> When providing VQNet code examples, you MUST first read the corresponding `references/*.md` file. **Never write VQNet API code from memory.**

### Why This Rule Exists
- VQNet API signatures are unique (e.g., `(input, param)` not `(param, input)`)
- dtype requirements differ from PyTorch (e.g., labels need `kint64`)
- Loss function order is reversed: `loss_fn(y_true, y_pred)` not `(y_pred, y_true)`
- VQC module requires `reset_states(batchsize)` per forward call

---

## When to Use This Skill

Use this skill when user mentions:
- VQNet, pyvqnet, QTensor
- QuantumLayer, VQC, pyqpanda3
- Quantum neural networks, variational quantum circuits
- Quantum machine learning, QVC, VSQL, Quanvolution
- Hybrid quantum-classical models
- GPU training with VQNet
- Distributed quantum ML training

---

## Reference Files Index

| 需求 | 读取文件 |
|------|----------|
| 安装/环境/FAQ | `references/install_env.md` |
| QTensor API | `references/qtensor.md` |
| 经典神经网络 (Module, Linear, Conv, Loss, Optimizer) | `references/classic_nn.md` |
| QuantumLayer (pyqpanda3) | `references/quantum_layers.md` |
| VQC 自动微分模块 | `references/vqc.md` |
| QML Demo 示例 | `references/qml_demos.md` |
| 分布式训练 | `references/distributed.md` |
| 量子大模型微调 | `references/quantum_llm.md` |

---

## Project Structure

```
vqnet2-skill/
├── SKILL.md                      # Core skill definition (分层体系)
├── CLAUDE.md                     # This file - AI assistant guide
├── README.md                     # Human-readable README
└── references/                   # API reference documentation
    ├── install_env.md           # 安装、环境配置、FAQ
    ├── qtensor.md               # QTensor 完整 API
    ├── classic_nn.md            # Module, Linear, Conv2D, Loss, Optimizer
    ├── quantum_layers.md        # QuantumLayer, QpandaQProgVQCLayer
    ├── vqc.md                   # VQC 自动微分 API
    ├── qml_demos.md             # QVC, QDRL, Quanvolution 示例
    ├── distributed.md           # MPI/NCCL 分布式训练
    └── quantum_llm.md           # 量子大模型微调
```

---

## Critical API Patterns

### 1. QuantumLayer 函数签名

```python
def circuit(input, param):  # 必须是 input 在前！
    # input: 经典输入数据
    # param: 变分参数
    return measurement_result  # np.ndarray 或 list
```

**常见错误**: 写成 `def circuit(param, input)` 会导致运行失败。

### 2. VQC 模块 reset_states

```python
from pyvqnet.qnn.vqc import QMachine, RZ, Probability

class QModel(Module):
    def forward(self, x):
        # 必须在 forward 开头调用！
        self.device.reset_states(x.shape[0])
        ...
```

### 3. 损失函数参数顺序

```python
loss_fn(y_true, y_pred)  # VQNet: 标签在前
loss_fn(y_pred, y_true)  # PyTorch: 预测在前
```

### 4. dtype 要求

| 用途 | dtype |
|------|-------|
| Embedding 输入 | `kint64` |
| CrossEntropyLoss 标签 | `kint64` |
| 普通张量 | `kfloat32`（默认） |

```python
from pyvqnet import kint64
labels = QTensor([0, 1, 2], dtype=kint64)
```

---

## Common Mistakes to Watch For

1. **参数顺序错误**: QuantumLayer 函数签名必须是 `(input, param)`
2. **忘记 reset_states**: VQC 模型每次 forward 必须调用
3. **dtype 错误**: CrossEntropy 标签必须是 `kint64`
4. **ModuleList vs list**: 子模块必须用 `ModuleList`
5. **QCloud Token**: 使用 `os.getenv("QCLOUD_TOKEN")`，不要硬编码
6. **混合后端**: 不同 backend 的 QTensor 不能混用
7. **GPU 训练**: 数据和模型都要移动到 GPU

---

## Dependencies

```bash
pip install pyvqnet
pip install pyqpanda3
# Optional:
pip install torch>=2.4.0,<2.7.0  # PyTorch backend
conda install conda-forge::mpich-mpicxx==4.1.2  # 分布式 CPU
pip install mpi4py  # 分布式 CPU
```

---

## Verification Checklist

Before generating VQNet code:

- [ ] 已读取对应的 references/*.md 文件
- [ ] QuantumLayer 函数签名 `(input, param)`
- [ ] VQC forward 开头调用 `reset_states(batchsize)`
- [ ] 损失函数 `(y_true, y_pred)` 顺序正确
- [ ] 标签 dtype 为 `kint64`
- [ ] 子模块使用 `ModuleList`
- [ ] 无硬编码 QCloud Token
- [ ] GPU 模型和数据都移动到 GPU

---

## Example: Quick Reference Lookup

```
User: "How to use QuantumLayer with pyqpanda3?"

AI Response Flow:
1. Read `references/quantum_layers.md`
2. Extract the example code
3. Verify function signature is (input, param)
4. Present code with import paths
```

---

## Additional Resources

- [Official VQNet Documentation](https://vqnet2-tutorial.readthedocs.io/)
- [Origin Quantum](https://www.originqc.com.cn/)
- [PyQPanda3 Documentation](https://qcloud.originqc.com.cn/document/qpanda-3/index.html)
- [Origin Quantum Cloud](https://qcloud.originqc.com.cn/)