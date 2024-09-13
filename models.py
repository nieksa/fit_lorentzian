import torch
import torch.nn as nn

class RegressionNet(nn.Module):
    """
    单纯的线性模型
    考虑添加dropout层和修改隐藏层来增强拟合能力防止过拟合
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionNet, self).__init__()
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
    单纯的线性模型
    考虑添加dropout层和修改隐藏层来增强拟合能力防止过拟合
    """
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(RegressionDropout, self).__init__()
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