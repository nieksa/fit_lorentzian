import torch
import torch.nn as nn
import torch.nn.functional as F

class RegressionNet(nn.Module):
    """
    单纯的线性模型
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionNet, self).__init__()
        self.name = 'RegressionNet'
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    

import torch
import torch.nn as nn

class RegressionDropout(nn.Module):
    """
    考虑添加dropout层和修改隐藏层来增强拟合能力防止过拟合
    """
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(RegressionDropout, self).__init__()
        self.name = 'RegressionDropout'
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        
        # 设置dropout率
        self.dropout_rate = dropout_rate
        # 添加dropout层
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # 在第一个全连接层后添加dropout
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)  # 在第二个全连接层后添加dropout
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)  # 在第三个全连接层后添加dropout
        x = torch.relu(self.fc4(x))
        x = self.dropout(x)  # 在第四个全连接层后添加dropout
        x = self.fc5(x)
        return x
    

class RegressionWithBatchNorm(nn.Module):
    """
    增强的线性模型，增加网络深度和批量归一化
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionWithBatchNorm, self).__init__()
        self.name = 'RegressionWithBatchNorm'
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # 添加批量归一化
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)  # 添加批量归一化
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)  # 添加批量归一化
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)  # 添加批量归一化
        self.fc5 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x


class CustomModel1(nn.Module):
    def __init__(self):
        super(CustomModel1, self).__init__()
        self.name = 'CustomModel1'
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 100, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CustomModel2(nn.Module):
    def __init__(self):
        super(CustomModel2, self).__init__()
        self.name = 'CustomModel2'
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=11, padding=5)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=15, padding=7)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 200, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
