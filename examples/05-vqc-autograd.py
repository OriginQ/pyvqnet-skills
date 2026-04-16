"""
VQC 自动微分模块示例
====================
展示 VQNet 内置的 VQC 模块（不依赖 pyqpanda）进行量子电路模拟和自动微分。

关键点：
- 必须在 forward 开头调用 device.reset_states(batchsize)
- 使用 QMachine 创建模拟器
- 量子门类：Hadamard, RX, RY, RZ, CNOT 等
"""

from pyvqnet.nn import Module, Linear
from pyvqnet.qnn.vqc import QMachine, RZ, RX, Probability
from pyvqnet.qnn.vqc.qcircuit import VQC_HardwareEfficientAnsatz
from pyvqnet.tensor import QTensor, arange
import pyvqnet


class QModel(Module):
    """混合量子经典模型"""

    def __init__(self):
        super().__init__()
        # 经典预处理层
        self.linear = Linear(4, 2)

        # 数据编码层（无参数）
        self.encode_z = RZ(wires=0)
        self.encode_x = RX(wires=1)

        # 可训练的量子门（设置 has_params=True, trainable=True）
        self.trainable_rx = RX(has_params=True, trainable=True, wires=0)
        self.trainable_rz = RZ(has_params=True, trainable=True, wires=1)

        # Ansatz 模板
        self.ansatz = VQC_HardwareEfficientAnsatz(
            num_wires=4,
            rot_gate_list=["rx", "RY", "rz"],
            entangle_gate="cnot",
            entangle_rules="linear",
            depth=2
        )

        # 测量层
        self.measure = Probability(wires=[0, 2])

        # 量子模拟器（4 个量子比特）
        self.device = QMachine(4)

    def forward(self, x):
        # 关键：必须重置状态！
        batchsize = x.shape[0]
        self.device.reset_states(batchsize)

        # 经典预处理
        y = self.linear(x)

        # 数据编码（参数来自经典层输出）
        # 注意：参数形状必须是 [batchsize, 1]
        self.encode_z(params=y[:, 0], q_machine=self.device)
        self.encode_x(params=y[:, 1], q_machine=self.device)

        # 可训练量子门（使用内部参数）
        self.trainable_rx(q_machine=self.device)
        self.trainable_rz(q_machine=self.device)

        # Ansatz 模板
        self.ansatz(q_machine=self.device)

        # 概率测量
        return self.measure(q_machine=self.device)


def main():
    print("=== VQC 自动微分示例 ===")
    print()

    # 创建模型
    model = QModel()
    print(f"模型参数数量: {len(model.parameters())}")
    print()

    # 创建输入数据
    batchsize = 3
    x = arange(1.0, batchsize * 4 + 1).reshape([batchsize, 4])
    x.requires_grad = True
    print(f"输入数据:\n{x}")
    print()

    # 前向传播
    print("=== 前向传播 ===")
    output = model(x)
    print(f"输出（测量概率）:\n{output}")
    print(f"输出形状: {output.shape}")
    print()

    # 反向传播
    print("=== 反向传播 ===")
    output.backward()

    # 检查输入梯度
    print(f"输入梯度:\n{x.grad}")
    print()

    # 检查量子参数梯度
    print(f"trainable_rx 参数梯度:\n{model.trainable_rx.params.grad}")
    print(f"trainable_rz 参数梯度:\n{model.trainable_rz.params.grad}")
    print()

    # 检查 Ansatz 参数梯度
    ansatz_params = model.ansatz.parameters()
    if ansatz_params and hasattr(ansatz_params[0], 'grad'):
        print(f"Ansatz 第一个参数梯度:\n{ansatz_params[0].grad}")
    print()

    print("示例完成！")
    print()
    print("关键要点总结：")
    print("1. 必须在 forward 开头调用 device.reset_states(batchsize)")
    print("2. 编码参数形状必须是 [batchsize, 1]")
    print("3. 可训练量子门设置 has_params=True, trainable=True")
    print("4. VQC 模块不依赖 pyqpanda，使用 VQNet 内置模拟")


if __name__ == "__main__":
    main()