"""
VQNet QTensor Basics Example
=============================
Demonstrates basic QTensor creation, attributes, and operations.
"""

from pyvqnet.tensor import QTensor, ones, randn, randu
from pyvqnet.dtype import *
import numpy as np

# 1. Create QTensor from list
print("=== 1. Creating QTensor from list ===")
t1 = QTensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
print(f"t1 = \n{t1}")
print()

# 2. Create QTensor from numpy
print("=== 2. Creating QTensor from numpy ===")
np_arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
t2 = QTensor(np_arr, dtype=kfloat64)
print(f"t2 shape: {t2.shape}")
print(f"t2 dtype: {t2.dtype}")
print(f"t2 = \n{t2}")
print()

# 3. Using factory functions
print("=== 3. Factory functions ===")
t_ones = ones([2, 3])
print(f"ones([2, 3]) =\n{t_ones}")
print()

# 4. Access attributes
print("=== 4. Attributes ===")
print(f"t1 ndim: {t1.ndim}")
print(f"t1 shape: {t1.shape}")
print(f"t1 size: {t1.size}")
print(f"t1 numel(): {t1.numel()}")
print(f"t1 dtype: {t1.dtype}")
print(f"t1 requires_grad: {t1.requires_grad}")
print()

# 5. Backward pass
print("=== 5. Backward propagation ===")
y = t1.sum()
y.backward()
print(f"t1.grad =\n{t1.grad}")
print()

# 6. Convert to numpy
print("=== 6. Convert to numpy ===")
np_result = t1.to_numpy()
print(f"type: {type(np_result)}")
print(f"value:\n{np_result}")
