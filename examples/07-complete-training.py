"""
完整训练循环示例
================
展示 VQNet 中完整的神经网络训练流程，包括：
- 数据加载
- 模型定义
- 损失函数（正确的参数顺序）
- 优化器
- 训练/验证循环
- 模型保存/加载

关键点：
- VQNet 损失函数参数顺序：(标签, 预测值)，与 PyTorch 相反
- CrossEntropy 标签需要 dtype=kint64
- 使用 optimizer._step() 而非 step()
"""

import numpy as np
from pyvqnet.nn import Module, Linear, ReLU, Sequential, Dropout
from pyvqnet.nn import CrossEntropyLoss, MeanSquaredError
from pyvqnet.optim import Adam, SGD
from pyvqnet.tensor import QTensor, ones, zeros, randn
from pyvqnet import kint64, kfloat32
from pyvqnet.utils import set_random_seed


# 设置随机种子
set_random_seed(42)


class MLPClassifier(Module):
    """多分类 MLP 模型"""

    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(),
            Dropout(0.2),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def generate_data(num_samples, input_dim, num_classes):
    """生成模拟分类数据"""
    np.random.seed(42)
    X = np.random.randn(num_samples, input_dim).astype(np.float32)

    # 生成标签（简单规则：根据第一维特征）
    y = np.zeros(num_samples, dtype=np.int64)
    for i in range(num_samples):
        if X[i, 0] > 0.5:
            y[i] = 0
        elif X[i, 0] < -0.5:
            y[i] = 1
        else:
            y[i] = 2

    return X, y


def data_generator(X, y, batch_size, shuffle=True):
    """批量数据生成器"""
    num_samples = X.shape[0]
    indices = np.arange(num_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        yield X[batch_indices], y[batch_indices]


def train_epoch(model, optimizer, loss_fn, X_train, y_train, batch_size):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y in data_generator(X_train, y_train, batch_size):
        # 转换为 QTensor
        # 注意：标签必须是 kint64 类型！
        x = QTensor(batch_x, requires_grad=True)
        y = QTensor(batch_y, dtype=kint64)  # 关键：CrossEntropy 标签需要 kint64

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        pred = model(x)

        # 计算损失
        # 注意：VQNet 损失函数参数顺序是 (标签, 预测值)
        loss = loss_fn(y, pred)  # 不是 loss_fn(pred, y)！

        # 反向传播
        loss.backward()

        # 参数更新
        optimizer._step()  # 注意：是 _step() 而非 step()

        # 统计
        total_loss += loss.item() * len(batch_y)

        # 计算准确率
        pred_np = pred.to_numpy()
        pred_labels = np.argmax(pred_np, axis=1)
        total_correct += np.sum(pred_labels == batch_y)
        total_samples += len(batch_y)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def evaluate(model, loss_fn, X_val, y_val, batch_size):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y in data_generator(X_val, y_val, batch_size, shuffle=False):
        x = QTensor(batch_x, requires_grad=False)
        y = QTensor(batch_y, dtype=kint64)

        pred = model(x)
        loss = loss_fn(y, pred)

        total_loss += loss.item() * len(batch_y)

        pred_np = pred.to_numpy()
        pred_labels = np.argmax(pred_np, axis=1)
        total_correct += np.sum(pred_labels == batch_y)
        total_samples += len(batch_y)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def main():
    print("=" * 60)
    print("VQNet 完整训练循环示例")
    print("=" * 60)
    print()

    # 配置
    input_dim = 8
    hidden_dim = 32
    num_classes = 3
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.01

    # 生成数据
    num_train = 500
    num_val = 100

    X_train, y_train = generate_data(num_train, input_dim, num_classes)
    X_val, y_val = generate_data(num_val, input_dim, num_classes)

    print(f"训练数据: {X_train.shape}, 标签: {y_train.shape}")
    print(f"验证数据: {X_val.shape}, 标签: {y_val.shape}")
    print(f"类别数: {num_classes}")
    print()

    # 创建模型
    model = MLPClassifier(input_dim, hidden_dim, num_classes)
    print(f"模型参数数量: {len(model.parameters())}")
    print()

    # 创建优化器和损失函数
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = CrossEntropyLoss()

    print("开始训练...")
    print("-" * 40)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc = train_epoch(
            model, optimizer, loss_fn,
            X_train, y_train, batch_size
        )

        # 验证
        val_loss, val_acc = evaluate(
            model, loss_fn,
            X_val, y_val, batch_size
        )

        # 打印结果
        print(f"Epoch {epoch + 1:3d}/{num_epochs}: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 注意：VQNet 模型保存需要手动保存参数
            # 这里仅作演示

    print("-" * 40)
    print()
    print(f"训练完成！最佳验证准确率: {best_val_acc:.4f}")
    print()

    # 测试单个样本
    print("=== 测试单个样本 ===")
    test_x = QTensor([[0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
    model.eval()
    pred = model(test_x)
    pred_label = np.argmax(pred.to_numpy())
    print(f"输入: {test_x.to_numpy()}")
    print(f"预测类别: {pred_label}")
    print(f"预测概率: {pred.to_numpy()}")
    print()

    # 关键要点总结
    print("=" * 60)
    print("关键要点总结")
    print("=" * 60)
    print()
    print("1. 损失函数参数顺序: loss_fn(y_true, y_pred)")
    print("   - VQNet 是 (标签, 预测值)")
    print("   - PyTorch 是 (预测值, 标签) - 注意区别！")
    print()
    print("2. CrossEntropy 标签 dtype 必须是 kint64")
    print("   y = QTensor(batch_y, dtype=kint64)")
    print()
    print("3. 优化器更新使用 optimizer._step()")
    print("   - 不是 step()，是 _step()")
    print()
    print("4. 训练前调用 optimizer.zero_grad()")
    print("5. 调用 loss.backward() 计算梯度")
    print("6. 训练时 model.train()，验证时 model.eval()")
    print()


if __name__ == "__main__":
    main()