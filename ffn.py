#%%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = torch.linspace(-np.pi, np.pi, 100).reshape(-1, 1)  # 输入数据：100个点，范围 [-π, π]
y = torch.sin(x)  # 目标函数：y = sin(x)

# 定义模型
class FFN(nn.Module):
    def __init__(self, hidden_dim):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(1, hidden_dim)  # 第一层：升维到 hidden_dim
        self.fc2 = nn.Linear(hidden_dim, 1)  # 第二层：降维回 1 维

    def forward(self, x):
        h = torch.relu(self.fc1(x))  # 第一层 + ReLU 激活函数
        y_pred = self.fc2(h)  # 第二层
        return y_pred, h

# 初始化模型
hidden_dim = 10  # 隐藏层维度
model = FFN(hidden_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred, _ = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 测试模型
with torch.no_grad():
    y_pred, h = model(x)

# 可视化结果
plt.figure(figsize=(15, 5))

# 1. 输入数据和目标函数
plt.subplot(1, 3, 1)
plt.scatter(x.numpy(), y.numpy(), label='True Data (y = sin(x))', color='blue')
plt.title('Input Data and Target Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# 2. 隐藏层的基函数
plt.subplot(1, 3, 2)
for i in range(hidden_dim):
    plt.plot(x.numpy(), h[:, i].numpy(), label=f'Basis Function {i+1}')
plt.title('Hidden Layer Basis Functions')
plt.xlabel('x')
plt.ylabel('h')
#plt.legend()

# 3. 拟合结果
plt.subplot(1, 3, 3)
plt.scatter(x.numpy(), y.numpy(), label='True Data (y = sin(x))', color='blue')
plt.plot(x.numpy(), y_pred.numpy(), label='FFN Prediction', color='red')
plt.title('FFN Fitting Result')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()
# %%
