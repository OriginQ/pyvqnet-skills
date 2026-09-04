# LLM 大模型算子与 TRL 微调损失

> **重要**: 所有示例代码均来自官方文档，可直接运行。

---

## 缩放 Softmax 函数族

`pyvqnet.nn.functional` 提供面向大模型注意力计算的缩放 softmax 函数族。

### scaled_softmax

```python
pyvqnet.nn.functional.scaled_softmax(input, scale: float)
```

对输入进行缩放 softmax 计算。

**参数**:
- `input` - 输入张量
- `scale` - 缩放系数

**返回**: 缩放后的 softmax 结果。

**示例**:
```python
from pyvqnet.tensor import QTensor, kfloat32
from pyvqnet.nn.functional import scaled_softmax

x = QTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=kfloat32)
out = scaled_softmax(x, 0.5)
print(out)

# [[0.1863237,0.3071959,0.5064804],
#  [0.1863237,0.3071959,0.5064804]]
```

### scaled_masked_softmax

```python
pyvqnet.nn.functional.scaled_masked_softmax(input, mask, scale: float)
```

带掩码的缩放 softmax 计算。

**参数**:
- `input` - 输入张量
- `mask` - 掩码张量，与 input 形状相同。`0`=保留，`1`=掩盖（对应位置被设为 -inf 后参与 softmax，结果为 0）
- `scale` - 缩放系数

**返回**: 缩放后的掩码 softmax 结果。

**示例**:
```python
from pyvqnet.tensor import QTensor, kfloat32
from pyvqnet.nn.functional import scaled_masked_softmax

x = QTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=kfloat32)
mask = QTensor([[0, 0, 1], [1, 0, 0]], dtype=kfloat32)  # 1=masked
out = scaled_masked_softmax(x, mask, 1.0)
print(out)

# [[0.2689414,0.7310586,0.       ],
#  [0.       ,0.2689414,0.7310586]]
```

### scaled_upper_triang_masked_softmax

```python
pyvqnet.nn.functional.scaled_upper_triang_masked_softmax(input, scale: float)
```

对输入上三角部分进行掩码的缩放 softmax 计算。

**参数**:
- `input` - 输入张量
- `scale` - 缩放系数

**返回**: 计算结果。

**示例**:
```python
from pyvqnet.tensor import QTensor, kfloat32
from pyvqnet.nn.functional import scaled_upper_triang_masked_softmax

x = QTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=kfloat32)
out = scaled_upper_triang_masked_softmax(x, 1.0)
print(out)

# [[1.       ,0.       ,0.       ],
#  [0.2689414,0.7310586,0.       ]]
```

---

## fused_moe 融合专家混合

```python
pyvqnet.nn.functional.fused_moe(input, fc1_weight, fc2_weight, probs, indices, num_experts)
```

融合专家混合 (MoE) 前向计算。

**参数**:
- `input` - 输入张量
- `fc1_weight` - 第一层专家权重，形状 `[E, H, 2F]`（示例中 E=2, H=4, F=4，即 `(2, 4, 8)`）
- `fc2_weight` - 第二层专家权重，形状 `[E, F, H]`（示例中为 `(2, 4, 4)`）
- `probs` - 专家路由概率
- `indices` - 专家索引
- `num_experts` - 专家数量

**返回**: MoE 计算的输出张量。

**注意**: **该函数仅在 GPU 下运行**。请将模型与数据转移至 GPU 后调用。

**示例**:
```python
import pyvqnet
from pyvqnet.tensor import QTensor, DEV_GPU_0, kfloat32
from pyvqnet.dtype import kint32
from pyvqnet.nn.functional import fused_moe
import numpy as np

np.random.seed(42)
E = 2
x = QTensor(np.random.randn(2, 4).astype(np.float32), device=DEV_GPU_0)
w1 = QTensor(np.random.randn(E, 4, 8).astype(np.float32), device=DEV_GPU_0)
w2 = QTensor(np.random.randn(E, 4, 4).astype(np.float32), device=DEV_GPU_0)
probs = QTensor([[0.7], [0.8]], dtype=kfloat32, device=DEV_GPU_0)
idx = QTensor([[0], [1]], dtype=kint32, device=DEV_GPU_0)
y = fused_moe(x, w1, w2, probs, idx, E)
print(y)

# [[ 0.4034894,-0.2737532,-1.0371764,-1.3686538],
#  [ 3.2209034, 0.9456251, 0.8900909, 2.3802228]]
```

---

## Token 采样函数族 (`pyvqnet.nn._sampling`)

LLM 推理中的 token 采样函数，位于 `pyvqnet.nn._sampling` 模块。

**重要约束**:
- **仅在 GPU (CUDA) 下运行**（官方文档对每个函数均注明"该函数仅在 GPU 下运行"）
- **纯推理操作**（无 autograd），不参与反向传播
- 阈值类参数接受**标量或 QTensor**（签名标注 `Union[QTensor, float]` / `Union[QTensor, int]`）
- 返回 `(tokens, valid_mask)` 元组：`tokens` 为采样词元索引，`valid_mask` 标记每个样本的采样是否有效

### top_k_top_p_sampling_from_logits

```python
pyvqnet.nn._sampling.top_k_top_p_sampling_from_logits(logits, temperature: Union[QTensor, float] = 1.0, top_k: Union[QTensor, int] = 0, top_p: Union[QTensor, float] = 1.0, deterministic: bool = True)
```

基于 logits 进行 top-k 与 top-p 联合采样。

**参数**:
- `logits` - 输入 logits 张量
- `temperature` - 温度系数，默认为 1.0
- `top_k` - 保留概率最高的 k 个词元，0 表示不限制，默认为 0
- `top_p` - 累积概率阈值，默认为 1.0（不限制）
- `deterministic` - 是否确定性采样，默认为 True

**注意**: 该函数仅在 GPU 下运行。

**示例**:
```python
import pyvqnet
from pyvqnet.tensor import QTensor, DEV_GPU_0, kfloat32
from pyvqnet.nn._sampling import top_k_top_p_sampling_from_logits

logits = QTensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
                 dtype=kfloat32, device=DEV_GPU_0)
tokens, valid = top_k_top_p_sampling_from_logits(logits, 1.0, 0, 1.0, True)
print(tokens)
# [3, 0]
print(valid)
# [ True, True]
```

### top_k_top_p_sampling_from_probs

```python
pyvqnet.nn._sampling.top_k_top_p_sampling_from_probs(probs, top_k: Union[QTensor, int] = 0, top_p: Union[QTensor, float] = 1.0, deterministic: bool = True)
```

基于概率分布进行 top-k 与 top-p 联合采样。

**参数**:
- `probs` - 输入概率张量
- `top_k` - 保留概率最高的 k 个词元，0 表示不限制，默认为 0
- `top_p` - 累积概率阈值，默认为 1.0
- `deterministic` - 是否确定性采样，默认为 True

**注意**: 该函数仅在 GPU 下运行。

**示例**:
```python
import pyvqnet
from pyvqnet.tensor import QTensor, DEV_GPU_0, kfloat32
from pyvqnet.nn._sampling import top_k_top_p_sampling_from_probs
from pyvqnet.utils import set_random_seed

set_random_seed(42)
probs = QTensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
                dtype=kfloat32, device=DEV_GPU_0)
tokens, valid = top_k_top_p_sampling_from_probs(probs, 0, 1.0, True)
print(tokens)
# [3,0]
print(valid)
# [ True, True]
```

### top_k_sampling_from_probs

```python
pyvqnet.nn._sampling.top_k_sampling_from_probs(probs, top_k: Union[QTensor, int] = 0, deterministic: bool = True)
```

基于概率分布进行 top-k 采样。

**参数**:
- `probs` - 输入概率张量
- `top_k` - 保留概率最高的 k 个词元，0 表示不限制，默认为 0
- `deterministic` - 是否确定性采样，默认为 True

**注意**: 该函数仅在 GPU 下运行。

**示例**:
```python
import pyvqnet
from pyvqnet.tensor import QTensor, DEV_GPU_0, kfloat32
from pyvqnet.nn._sampling import top_k_sampling_from_probs
from pyvqnet.utils import set_random_seed

set_random_seed(42)
probs = QTensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
                dtype=kfloat32, device=DEV_GPU_0)
tokens, valid = top_k_sampling_from_probs(probs, 0, True)
print(tokens)
# [3,0]
print(valid)
# [ True, True]
```

### top_p_sampling_from_probs

```python
pyvqnet.nn._sampling.top_p_sampling_from_probs(probs, top_p: Union[QTensor, float] = 1.0, deterministic: bool = True)
```

基于概率分布进行 top-p 采样。

**参数**:
- `probs` - 输入概率张量
- `top_p` - 累积概率阈值，默认为 1.0
- `deterministic` - 是否确定性采样，默认为 True

**注意**: 该函数仅在 GPU 下运行。

**示例**:
```python
import pyvqnet
from pyvqnet.tensor import QTensor, DEV_GPU_0, kfloat32
from pyvqnet.nn._sampling import top_p_sampling_from_probs
from pyvqnet.utils import set_random_seed

set_random_seed(42)
probs = QTensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
                dtype=kfloat32, device=DEV_GPU_0)
tokens, valid = top_p_sampling_from_probs(probs, 1.0, True)
print(tokens)
# [3,0]
print(valid)
# [ True, True]
```

### min_p_sampling_from_probs

```python
pyvqnet.nn._sampling.min_p_sampling_from_probs(probs, min_p: Union[QTensor, float] = 0.0, deterministic: bool = True)
```

基于 min-p 策略的概率采样。

**参数**:
- `probs` - 输入概率张量
- `min_p` - 最小概率阈值，默认为 0.0
- `deterministic` - 是否确定性采样，默认为 True

**注意**: 该函数仅在 GPU 下运行。

**示例**:
```python
import pyvqnet
from pyvqnet.tensor import QTensor, DEV_GPU_0, kfloat32
from pyvqnet.nn._sampling import min_p_sampling_from_probs
from pyvqnet.utils import set_random_seed

set_random_seed(42)
probs = QTensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
                dtype=kfloat32, device=DEV_GPU_0)
tokens, valid = min_p_sampling_from_probs(probs, 0.0, True)
print(tokens)
# [3,0]
print(valid)
# [ True, True]
```

---

## `pyvqnet.torch.trl` 大模型微调损失

以下损失函数用于基于强化学习/偏好的大模型微调 (RLHF/DPO/PPO/GRPO/SFT)，位于 `pyvqnet.torch.trl` 模块。模型为 `torch.nn.Module`（前向传播返回 logits），输入输出均为 torch 张量。

### sft_loss

```python
pyvqnet.torch.trl.sft_loss(model, input_ids, labels, ignore_index=-100)
```

监督式微调 (SFT) 交叉熵损失。

**参数**:
- `model` - 神经网络模块，前向传播返回 logits (B, L, V)
- `input_ids` - 输入 token ID 序列 (B, L)
- `labels` - 目标 token ID 标签 (B, L)
- `ignore_index` - 忽略的标签索引，默认为 -100

**返回**: SFT 损失值。

**示例**:
```python
import torch
torch.manual_seed(42)
import torch.nn as nn
from pyvqnet.torch.trl import sft_loss

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 8)
        self.head = nn.Linear(8, 8)
    def forward(self, x):
        return self.head(self.embed(x))

loss = sft_loss(TinyLM(), torch.tensor([[1,2,3,4]]), torch.tensor([[2,3,4,5]]))
print(loss.item())
# 2.1718
```

### dpo_loss

```python
pyvqnet.torch.trl.dpo_loss(policy_model, ref_model, chosen_ids, rejected_ids, chosen_mask, rejected_mask, beta=0.1)
```

DPO (Direct Preference Optimization) 标准 sigmoid 偏好损失。通过最大化偏好与非偏好序列之间的隐式奖励差异进行优化。

**参数**:
- `policy_model` - 策略网络，前向传播返回 logits (B, L, V)
- `ref_model` - 参考网络，前向传播返回 logits (B, L, V)
- `chosen_ids` - 偏好序列的 token ID (B, L_chosen)
- `rejected_ids` - 非偏好序列的 token ID (B, L_rejected)
- `chosen_mask` - 偏好序列的注意力掩码
- `rejected_mask` - 非偏好序列的注意力掩码
- `beta` - KL 正则化系数，默认为 0.1

**返回**: DPO 损失值。

**示例**:
```python
import torch
torch.manual_seed(42)
import torch.nn as nn
from pyvqnet.torch.trl import dpo_loss

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 8)
        self.head = nn.Linear(8, 8)
    def forward(self, x):
        return self.head(self.embed(x))

policy, ref = TinyLM(), TinyLM()
loss = dpo_loss(policy, ref,
    torch.tensor([[0,1,2,3,4,5]]), torch.tensor([[5,4,3,2,1,0]]),
    torch.tensor([[1,1,1,1,1,1]]), torch.tensor([[1,1,1,1,1,1]]),
    beta=0.1)
print(loss.item())
# 0.7008
```

### ppo_loss

```python
pyvqnet.torch.trl.ppo_loss(policy_model, value_model, ref_model, query_responses, context_length, response_mask, old_logprobs, old_values, advantages, returns, cliprange=0.2, cliprange_value=0.2, vf_coef=1.0, temperature=1.0)
```

PPO (Proximal Policy Optimization) 策略与价值函数联合损失。包含裁剪的替代策略损失和价值函数损失（配合优势/回报估计，即 GAE 输出，进行策略与价值的双重裁剪）。

**参数**:
- `policy_model` - 策略网络，前向传播返回 logits
- `value_model` - 价值网络，前向传播返回标量值
- `ref_model` - 参考网络，用于 KL 惩罚
- `query_responses` - 查询与响应拼接的 token ID 序列 (B, L)
- `context_length` - 查询部分的长度，用于区分查询与响应
- `response_mask` - 响应部分掩码 (B, L)，1=响应 token
- `old_logprobs` - 旧策略下的对数概率
- `old_values` - 旧价值网络的估计值
- `advantages` - 优势函数估计
- `returns` - 折扣回报
- `cliprange` - 策略裁剪范围，默认为 0.2
- `cliprange_value` - 价值函数裁剪范围，默认为 0.2
- `vf_coef` - 价值函数损失系数，默认为 1.0
- `temperature` - 采样温度，默认为 1.0

**返回**: PPO 损失值。

**示例**:
```python
import torch
torch.manual_seed(42)
import torch.nn as nn
from pyvqnet.torch.trl import ppo_loss

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 8)
        self.head = nn.Linear(8, 8)
    def forward(self, x):
        return self.head(self.embed(x))

class TinyValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 8)
        self.head = nn.Linear(8, 1)
    def forward(self, x):
        return self.head(self.embed(x)).squeeze(-1)

loss = ppo_loss(TinyLM(), TinyValue(), TinyLM(),
    torch.tensor([[0,1,2,3,4,5,6,7]]), 2,
    torch.ones(1,6), torch.zeros(1,6), torch.zeros(1,6),
    torch.ones(1,6), torch.ones(1,6))
print(loss.item())
# 1.3638
```

### grpo_loss

```python
pyvqnet.torch.trl.grpo_loss(policy_model, ref_model, input_ids, completion_mask, old_per_token_logps, advantages, beta=0.0, epsilon=0.2, epsilon_low=None, epsilon_high=None)
```

GRPO (Group Relative Policy Optimization) 裁剪替代损失。将多个补全结果分组计算优势（组归一化优势：优势由外部按组计算后经 `advantages` 参数传入）。

**参数**:
- `policy_model` - 策略网络，前向传播返回 logits
- `ref_model` - 参考网络或 None
- `input_ids` - 提示与补全拼接的 token ID 序列 (B*G, L)，其中 G 为组大小
- `completion_mask` - 补全部分掩码 (B*G, L)，1=补全 token，0=提示/填充
- `old_per_token_logps` - 旧策略下的逐 token 对数概率 (B*G, T)
- `advantages` - 组内的优势函数估计
- `beta` - KL 惩罚系数，默认为 0.0
- `epsilon` - PPO 裁剪范围，默认为 0.2
- `epsilon_low` - 裁剪下限，默认为 None（使用 epsilon）
- `epsilon_high` - 裁剪上限，默认为 None（使用 epsilon）

**返回**: GRPO 损失值。

**示例**:
```python
import torch
torch.manual_seed(42)
import torch.nn as nn
from pyvqnet.torch.trl import grpo_loss

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 8)
        self.head = nn.Linear(8, 8)
    def forward(self, x):
        return self.head(self.embed(x))

loss = grpo_loss(TinyLM(), TinyLM(),
    torch.tensor([[0,1,2,3,4],[5,6,7,0,1]]),
    torch.tensor([[0,0,1,1,1],[0,1,1,1,1]]),
    torch.zeros(2,3), torch.tensor([1.0, -0.5]),
    beta=0.0, epsilon=0.2)
print(loss.item())
# 0.1405
```

### reward_loss

```python
pyvqnet.torch.trl.reward_loss(model, chosen_ids, rejected_ids, margin=None, center_coef=None)
```

奖励模型对比损失（Bradley-Terry 风格）。通过最大化偏好与非偏好序列之间的奖励差异训练奖励模型。

**参数**:
- `model` - 奖励模型，前向传播返回标量奖励值
- `chosen_ids` - 偏好序列的 token ID
- `rejected_ids` - 非偏好序列的 token ID
- `margin` - 对比间隔，默认为 None
- `center_coef` - 奖励中心化系数，默认为 None

**返回**: 奖励模型损失值。

**示例**:
```python
import torch
torch.manual_seed(42)
import torch.nn as nn
from pyvqnet.torch.trl import reward_loss

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 8)
        self.head = nn.Linear(8, 8)
    def forward(self, x):
        return self.head(self.embed(x))

loss = reward_loss(TinyLM(), torch.tensor([[1,2,3]]), torch.tensor([[3,2,1]]))
print(loss.item())
# 0.7172
```

---

## 常见问题

1. **CUDA-only 算子**: `fused_moe` 与 `pyvqnet.nn._sampling` 全部采样函数**仅在 GPU 下运行**，数据需以 `device=DEV_GPU_0` 创建
2. **采样为纯推理**: 采样函数无 autograd，不参与训练；`fused_moe` 用于 MoE 前向计算
3. **采样阈值参数**: `top_k`/`top_p`/`min_p`/`temperature` 可传标量或 `QTensor`（`Union[QTensor, int|float]`）
4. **采样返回值**: 返回 `(tokens, valid_mask)` 元组，需解包接收
5. **scaled_masked_softmax 掩码语义**: `0`=保留，`1`=掩盖（掩盖位置输出为 0）
6. **trl 损失基于 torch 后端**: 模型为 `torch.nn.Module`（forward 返回 logits `(B, L, V)`），输入为 `torch.tensor`，而非 QTensor
7. **sft_loss 忽略标签**: `ignore_index=-100` 为默认忽略索引，与 HF 惯例一致
8. **grpo_loss 优势来源**: 组归一化优势在调用方计算，经 `advantages` 参数传入；`ref_model` 可为 None

---

**Version**: VQNet 2.18.1
