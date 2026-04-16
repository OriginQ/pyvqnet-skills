"""
GPU 训练示例
============
展示如何在 VQNet 中使用 GPU 进行神经网络训练。

关键点：
- 模型需要调用 .toGPU() 移动到 GPU
- 数据需要使用 device=DEV_GPU_0 或 .toGPU()
- CUDA 架构要求：sm_80 (A100) 或 sm_86 (RTX 30系列)

注意：运行此示例需要支持 CUDA 的 GPU
"""

from pyvqnet.nn import Module, Linear, ReLU, Sequential
from pyvqnet.nn import MeanSquaredError
from pyvqnet.optim import Adam
from pyvqnet.tensor import QTensor, randn
from pyvqnet import DEV_GPU_0, DEV_CPU
import pyvqnet


class SimpleMLP(Module):
    """简单的 MLP 模型"""

    def __init__(self):
        super().__init__()
        self.net = Sequential(
            Linear(10, 32),
            ReLU(),
            Linear(32, 16),
            ReLU(),
            Linear(16, 5)
        )

    def forward(self, x):
        return self.net(x)


def check_gpu_available():
    """检查 GPU 是否可用"""
    try:
        test = QTensor([1.0], device=DEV_GPU_0)
        return True
    except Exception as e:
        print(f"GPU 不可用: {e}")
        return False


def train_on_gpu():
    """GPU 训练流程"""
    print("=== GPU 训练示例 ===")
    print()

    # 检查 GPU
    if not check_gpu_available():
        print("GPU 不可用，请确保：")
        print("1. 安装了支持 CUDA 的 GPU")
        print("2. GPU 架构为 sm_80 或 sm_86")
        print("运行 CPU 版本演示...")
        train_on_cpu()
        return

    print("GPU 可用，开始训练...")
    print()

    # 创建模型并移动到 GPU
    model = SimpleMLP()
    model.toGPU(DEV_GPU_0)
    print(f"模型已移动到 GPU (device={DEV_GPU_0})")
    print()

    # 创建优化器和损失函数
    optimizer = Adam(model.parameters(), lr=0.01)
    loss_fn = MeanSquaredError()

    # 创建 GPU 数据
    batch_size = 32
    x = randn([batch_size, 10], device=DEV_GPU_0)
    x.requires_grad = True
    y = randn([batch_size, 5], device=DEV_GPU_0)

    print(f"数据已创建在 GPU 上")
    print(f"输入形状: {x.shape}")
    print(f"目标形状: {y.shape}")
    print()

    # 训练循环
    num_epochs = 5
    for epoch in range(num_epochs):
        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        pred = model(x)

        # 计算损失（注意 VQNet 损失函数顺序：标签在前，预测在后）
        loss = loss_fn(y, pred)

        # 反向传播
        loss.backward()

        # 参数更新
        optimizer._step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")

    print()
    print("GPU 训练完成！")
    print()
    print("关键要点：")
    print("1. 模型移动到 GPU: model.toGPU(DEV_GPU_0)")
    print("2. 数据创建在 GPU: randn([shape], device=DEV_GPU_0)")
    print("3. 或数据移动到 GPU: tensor.toGPU()")
    print("4. 模型和数据必须在同一设备上")


def train_on_cpu():
    """CPU 训练流程（作为对比）"""
    print("=== CPU 训练示例 ===")
    print()

    # 创建模型
    model = SimpleMLP()
    print("模型在 CPU 上")
    print()

    # 创建优化器和损失函数
    optimizer = Adam(model.parameters(), lr=0.01)
    loss_fn = MeanSquaredError()

    # 创建 CPU 数据
    batch_size = 32
    x = randn([batch_size, 10])  # 默认 device=DEV_CPU
    x.requires_grad = True
    y = randn([batch_size, 5])

    # 训练循环
    num_epochs = 3
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(y, pred)
        loss.backward()
        optimizer._step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")

    print()
    print("CPU 训练完成！")


def gpu_tensor_operations():
    """GPU 张量操作示例"""
    print("=== GPU 张量操作 ===")
    print()

    if not check_gpu_available():
        return

    # 在 GPU 上创建张量
    a = QTensor([[1.0, 2.0], [3.0, 4.0]], device=DEV_GPU_0)
    b = QTensor([[5.0, 6.0], [7.0, 8.0]], device=DEV_GPU_0)

    print(f"a.device = {a.device} (DEV_GPU_0 = {DEV_GPU_0})")
    print(f"b.device = {b.device}")
    print()

    # GPU 上的运算
    c = a + b
    print(f"a + b =\n{c.to_numpy()}")  # 转换回 CPU 显示

    # 移动回 CPU
    a_cpu = a.CPU()
    print(f"a_cpu.device = {a_cpu.device} (DEV_CPU = {DEV_CPU})")
    print()


def main():
    print("VQNet GPU 训练示例")
    print("=" * 50)
    print()

    # GPU 训练
    train_on_gpu()
    print()

    # GPU 张量操作
    gpu_tensor_operations()

    print("=" * 50)
    print("示例完成")


if __name__ == "__main__":
    main()