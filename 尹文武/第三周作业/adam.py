import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(1)
np.random.seed(1)

# print(torch.randn(2,3))
# print(np.random.randn(2,3))

LR      = 1e-3
BETA1   = 0.9
BETA2   = 0.999
EPSILON = 1e-8

# 简单线性层
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(64, 64)

    def forward(self, data):
        return self.fc1(data)

x           = torch.randn(2, 64)
target      = torch.randn(2, 64)
net_torch   = SimpleNet()
weight_init = net_torch.fc1.weight.data.clone()
# print(weight_init)

optimizer   = torch.optim.Adam(net_torch.parameters(), lr      = LR, betas = (BETA1, BETA2), eps = EPSILON)
loss_func   = nn.MSELoss()
pred        = net_torch.forward(x)
loss        = loss_func(pred, target)
# print(loss)
optimizer.zero_grad()
loss.backward()

# 拿到梯度（反向传播之后、step 之前）
grad_torch = net_torch.fc1.weight.grad.clone()

optimizer.step()

W_after_torch = net_torch.fc1.weight.data.clone()

# print("=" * 60)
# print("【PyTorch Adam】")
# print(f"  loss          : {loss.item():.6f}")
# print(f"  weight (init) :\n{weight_init.numpy()}")
# print(f"  grad          :\n{grad_torch.numpy()}")
# print(f"  weight (after):\n{W_after_torch.numpy()}")

# 从相同初始权重出发，使用相同梯度
W_manual  = weight_init.numpy().copy()
grad      = grad_torch.numpy().copy() # 梯度
# print(weight_init)
# print(weight_init.numpy())

t  = 0
mt = np.zeros_like(W_manual) # 一阶矩（动量）
vt = np.zeros_like(W_manual) # 二阶矩（梯度平方）
gt = grad

# 跟新一步
t        = t + 1
mt       = BETA1 * mt + (1 - BETA1) * gt
vt       = BETA2 * vt + (1 - BETA2) * gt**2

mth      = mt / (1 - BETA1 ** t)
vth      = vt / (1 - BETA2 ** t)

W_manual = W_manual - LR * mth / (np.sqrt(vth) + EPSILON)

# print()
# print("=" * 60)
# print("【手动 Adam】")
# print(f"  t             : {t}")
# print(f"  mt (一阶矩)   :\n{mt}")
# print(f"  vt (二阶矩)   :\n{vt}")
# print(f"  m̂t (修正后)   :\n{mth}")
# print(f"  v̂t (修正后)   :\n{vth}")
# print(f"  weight (after):\n{W_manual}")

# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — 对比
# ══════════════════════════════════════════════════════════════════════════════
diff = np.abs(W_after_torch.numpy() - W_manual)
print()
print("=" * 60)
print("【对比】权重差异（手动 vs PyTorch）")
print(f"  最大误差: {diff.max():.2e}")
print(f"  平均误差: {diff.mean():.2e}")
print("  结论:", "✓ 完全一致（误差在浮点精度范围内）"
               if diff.max() < 1e-6 else "✗ 存在较大差异，请检查")