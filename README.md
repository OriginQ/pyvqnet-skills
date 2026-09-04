# VQNet 技能库使用指南

## VQNet 技能库介绍

### 什么是 VQNet 技能库

VQNet 技能库是一个 AI 辅助量子机器学习编程技能库，是本源量子 AI 辅助量子编程套件的工具之一。它与 pyvqnet（量子机器学习编程库）和 pyqpanda3（量子计算编程库）协同工作，为用户提供自然语言驱动的量子机器学习编程体验。

你只需用自然语言描述量子机器学习任务，AI 即可理解意图并生成对应的 pyvqnet 代码，完成量子线路构建、混合量子经典模型搭建、模型训练与调优、分布式训练、结果分析等全流程，支持本地虚拟机模拟执行和提交本源量子云真机执行任务。

### 技能库的能力

通过加载 VQNet 技能库，AI 能够帮助你完成以下工作：

- **量子张量计算**：QTensor 自动微分张量、dtype 类型体系、GPU 加速、CPU/GPU 设备管理
- **经典神经网络**：Module、Linear、Conv2D、BatchNorm、LSTM 等网络层，损失函数与优化器，模型定义方式与主流机器学习框架一致
- **量子线路构建**：QMachine 量子虚拟机，Hadamard、RX、RY、RZ、CNOT 等量子门操作，复杂线路组合与测量操作
- **量子层封装**：QuantumLayer、QuantumBatchAsyncQcloudLayer（云真机异步）、NoiseQuantumLayer（噪声模拟层）、QuantumLayerAdjoint（伴随法梯度）
- **变分量子线路（VQC）**：内置自动微分模拟、参数化线路构建、硬件高效 ansatz、振幅/角度嵌入模板、量子自然梯度（QNG）、QNSPSA 优化器
- **量子信息分析**：概率测量、期望值计算、量子熵、互信息等测量与熵分析
- **QML 算法应用**：量子分类器、QVC、QDRL、Quanvolution、量子经典混合模型（CNN + QNN）
- **分布式训练**：MPI 多进程 CPU 训练、NCCL GPU 通信、多节点部署、流水线并行、张量并行、ZeRO 零冗余优化器
- **PyTorch 后端**：无缝切换 torch 计算后端；torch 后端状态向量量子线路（`sv.torch`）与张量网络量子线路（`tn.torch`）
- **大模型算子与微调**：scaled softmax 族、top-k/top-p/min-p token 采样族、fused_moe 混合专家算子，trl 微调损失（SFT / DPO / PPO / GRPO / Reward）
- **量子大模型**：quantum-llm 结合 Llama Factory，使用 VQC 进行大模型微调
- **量子真机使用**：将量子线路提交到本源量子云真机运行

### 工作方式

本技能库以 SKILL.md 技能索引 + references/ 官方 API 参考的形式加载到 AI 助手（Claude Code、Cline 等），提供：

- **API 速查** - 快速访问函数签名、参数与官方示例
- **代码生成** - 按官方文档惯例生成正确的 VQNet 代码
- **调试辅助** - 识别常见错误（参数顺序、dtype、reset_states 遗漏等）
- **最佳实践** - 遵循官方文档约定

## Supported APIs

| Category | Key APIs |
|----------|----------|
| **QTensor** | Core tensor with automatic differentiation, GPU support |
| **Classical NN** | Module, Linear, Conv2D, BatchNorm, LSTM, Loss, Optimizer |
| **QuantumLayer (pyqpanda3)** | QuantumLayer, QuantumBatchAsyncQcloudLayer, QuantumLayerAdjoint |
| **VQC Autograd** | QMachine, Hadamard/RX/RY/RZ/CNOT, Probability, reset_states |
| **Hybrid Layers** | QLinear (quantum FC), QConv (quantum convolution) |
| **Templates** | HardwareEfficientAnsatz, AmplitudeEmbedding, AngleEmbedding |
| **Distributed** | MPI/NCCL multi-GPU training (Linux only) |
| **Torch Quantum Circuits** | `sv.torch`（状态向量 36 门）、`tn.torch`（张量网络）、`pq3.torch` 量子层 |
| **LLM Ops & TRL** | scaled_softmax 族、top-k/top-p/min-p 采样、fused_moe、sft/dpo/ppo/grpo/reward_loss |
| **Quantum LLM** | Fine-tuning with quantum circuits via quantum-llm |

## Installation

### 安装方法

将仓库克隆到本地，然后把技能目录复制（或软链接）到你所用的 AI 工具对应技能目录即可：

```bash
git clone https://github.com/OriginQ/pyvqnet-skills.git
```

**技能目录对照表**：

| 工具 | 用户/全局技能目录 | 项目技能目录 |
|------|------------------|--------------|
| OpenCode | `~/.config/opencode/skills/` | `<project>/.opencode/skills/` |
| Claude Code | `~/.claude/skills/` | `<project>/.claude/skills/` |
| Gemini CLI | `~/.gemini/skills/` | `<project>/.gemini/skills/` |
| Codex | `~/.codex/skills/` | `<project>/.codex/skills/` |
| Cline | `~/.cline/skills/` | 仅全局 |

以 Claude Code 全局安装为例：

```bash
# Copy to Claude Code skills directory
cp -r pyvqnet-skills ~/.claude/skills/vqnet2-api
```

其他工具同理：将仓库复制到上表对应的技能目录下即可，目录名即技能加载名。

### Requirements for VQNet Development

```bash
pip install pyvqnet
pip install pyqpanda3

# Optional:
pip install torch                     # PyTorch backend（需自行安装，官方基准 torch 2.11.0+cu126 验证）
conda install conda-forge::mpich-mpicxx==4.1.2  # Distributed CPU
pip install mpi4py                  # Distributed CPU
```

## Usage

### Automatic Triggering

This skill triggers when you mention:
- VQNet, pyvqnet, QTensor
- QuantumLayer, VQC, pyqpanda3
- Quantum neural networks, variational quantum circuits
- Quantum machine learning, QVC, VSQL, Quanvolution

### Example: QuantumLayer

```python
from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
from pyvqnet.qnn.pq3.measure import ProbsMeasure
import pyqpanda3.core as pq

def circuit(input, param):  # 注意：input 在前，param 在后
    machine = pq.CPUQVM()
    qubits = range(4)
    cir = pq.QCircuit()

    for q, x in zip(qubits, input):
        cir << pq.H(q) << pq.RZ(q, x)

    for i in range(3):
        cir << pq.CNOT(i, i + 1) << pq.RY(i + 1, param[i])

    prog = pq.QProg()
    prog << cir
    return ProbsMeasure(machine, prog, list(qubits))

layer = QuantumLayer(circuit, 3)  # 3 个可训练参数
```

### Example: VQC Autograd Module

```python
from pyvqnet.qnn.vqc import QMachine, RZ, Probability
from pyvqnet.nn import Module, Linear

class QModel(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(4, 2)
        self.encode = RZ(wires=0)
        self.device = QMachine(4)

    def forward(self, x):
        # 必须在 forward 开头调用！
        self.device.reset_states(x.shape[0])
        y = self.linear(x)
        self.encode(params=y[:, 0], q_machine=self.device)
        return Probability(wires=[0])(q_machine=self.device)
```

### Example: Complete Training

```python
from pyvqnet.nn import Module, Linear, CrossEntropyLoss
from pyvqnet.optim import Adam
from pyvqnet.tensor import QTensor
from pyvqnet import kint64

model = Linear(10, 5)
optimizer = Adam(model.parameters(), lr=0.01)
loss_fn = CrossEntropyLoss()

x = QTensor([[0.1, ...]], requires_grad=True)
y = QTensor([0], dtype=kint64)  # 标签必须是 kint64

pred = model(x)
loss = loss_fn(y, pred)  # 注意：VQNet 是 (标签, 预测值)

optimizer.zero_grad()
loss.backward()
optimizer._step()  # 注意：是 _step() 而非 step()
```

## Critical API Patterns

| Pattern | VQNet | PyTorch (对比) |
|---------|--------|----------------|
| QuantumLayer 签名 | `(input, param)` | - |
| 损失函数参数 | `(y_true, y_pred)` | `(y_pred, y_true)` |
| CrossEntropy 标签 dtype | `kint64` | `torch.long` |
| 优化器更新 | `optimizer._step()` | `optimizer.step()` |
| VQC forward | 必须调用 `reset_states(batchsize)` | - |

## Project Structure

```
vqnet2-skill/
├── SKILL.md                      # Core skill - 12-layer skill system
├── CLAUDE.md                     # AI assistant guide
├── README.md                     # This file
├── examples/                     # Working example scripts
│   ├── 01-qtensor-basics.py
│   ├── 02-quantum-layer.py
│   ├── 03-qnn-classification.py
│   ├── 04-pytorch-backend.py
│   ├── 05-vqc-autodiff.py        # VQC reset_states 示例
│   ├── 06-gpu-training.py        # GPU toGPU() 示例
│   └── 07-complete-training.py   # 完整训练循环
└── references/                   # API reference (from official RST)
    ├── install_env.md            # 安装 + FAQ
    ├── qtensor.md                # QTensor API
    ├── classic_nn.md             # Module, Linear, Conv, Loss, Optimizer
    ├── utils.md                  # 神经网络工具
    ├── quantum_layers.md         # QuantumLayer, QcloudLayer
    ├── vqc.md                    # VQC autograd module
    ├── measurement.md            # 量子测量与熵
    ├── qnn.md                    # QNN 架构总览
    ├── quantum_templates.md      # 电路模板与拟设
    ├── qml_demos.md              # QVC, QDRL, Quanvolution
    ├── distributed.md            # MPI/NCCL 分布式
    ├── quantum_llm.md            # Quantum LLM fine-tuning
    ├── torch_api.md              # PyTorch 后端 / sv.torch / tn.torch
    └── llm_ops.md                # 大模型算子 + trl 微调损失
```

## Common Mistakes Prevented

| Mistake | Fix |
|---------|-----|
| `def circuit(param, input)` | Use `(input, param)` |
| `loss_fn(pred, y)` | Use `loss_fn(y, pred)` |
| Labels as `kfloat32` | Use `kint64` for CrossEntropy |
| `optimizer.step()` | Use `optimizer._step()` |
| Missing `reset_states` | Call `device.reset_states(batchsize)` in forward |
| Python `list` for submodules | Use `ModuleList` |
| Hardcoded QCloud token | Use `os.getenv("QCLOUD_TOKEN")` |

## Links

- [Official VQNet Documentation](https://vqnet2-tutorial.readthedocs.io/)
- [Origin Quantum](https://www.originqc.com.cn/)
- [PyQPanda3 Documentation](https://qcloud.originqc.com.cn/document/qpanda-3/index.html)
- [Origin Quantum Cloud](https://qcloud.originqc.com.cn/)

## License

Apache License 2.0

## Credits

Based on official VQNet 2.0 documentation from Origin Quantum.