# %%
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import EarlyStopping
from models import RegressionNet, RegressionDropout
from generator import generate_data

# %%
"""
构建网络 创建model时候指定参数
选择损失函数和优化器
"""
input_size = 400
hidden_size = 128
output_size = 5
model = RegressionNet(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# %%
"""提交到服务器前修改超参数"""
sample_nums = 100000
num_epochs = 400
batch_size = 64
patience =5

# %%
"""
除非对数据集划分比例有要求否则无需更改
"""
x_data, y_data = generate_data(sample_nums)
input_data = y_data
output_data = x_data

X_train, X_temp, y_train, y_temp = train_test_split(input_data, output_data, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
"""
除非网络对输入有特殊要求非则无需改动
"""

early_stopping = EarlyStopping(patience=patience, verbose=True)
stopped_epoch = None
best_val_loss = float('inf')
best_model_state = None
print(f"training with {device}")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        train_loss += loss.item() * batch_x.size(0)
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader.dataset)
    model.eval()
    val_loss = 0
    test_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_x.size(0)

        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_x.size(0)

    val_loss /= len(val_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss : {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    print(f'Test Loss:{test_loss}')
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch:{epoch}")
        stopped_epoch = epoch
        break
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        best_model_epoch = epoch


torch.save(best_model_state, f'save_best_model_epoch_{best_model_epoch}.pth')
if early_stopping.early_stop:
    torch.save(model.state_dict(), f'save_stopped_epoch_{stopped_epoch}.pth')
else:
    torch.save(model.state_dict(), f'save_last_model_{num_epochs}.pth')


