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
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out
        
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, output_size=5, zero_init_residual=False):
        super(ResNet1D, self).__init__()
        self.name = 'ResNet1D'
        self.in_planes = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(512 * block.expansion, output_size)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    # def get_feat_modules(self):
    #     feat_m = nn.ModuleList([])
    #     feat_m.append(self.conv1)
    #     feat_m.append(self.bn1)
    #     feat_m.append(self.layer1)
    #     feat_m.append(self.layer2)
    #     feat_m.append(self.layer3)
    #     feat_m.append(self.layer4)
    #     return feat_m
    
    # def get_bn_before_relu(self):
    #     if isinstance(self.layer1[0], Bottleneck):
    #         bn1 = self.layer1[-1].bn3
    #         bn2 = self.layer2[-1].bn3
    #         bn3 = self.layer3[-1].bn3
    #         bn4 = self.layer4[-1].bn3
    #     elif isinstance(self.layer1[0], BasicBlock):
    #         bn1 = self.layer1[-1].bn2
    #         bn2 = self.layer2[-1].bn2
    #         bn3 = self.layer3[-1].bn2
    #         bn4 = self.layer4[-1].bn2
    #     else:
    #         raise NotImplementedError('ResNet unknown block error !!!')

    #     return [bn1, bn2, bn3, bn4]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride, i == num_blocks - 1))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for 1D input
        out = F.relu(self.bn1(self.conv1(x)))
        out, _ = self.layer1(out)
        out, _ = self.layer2(out)
        out, _ = self.layer3(out)
        out, _ = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        
def ResNet50_1D(**kwargs):
    return ResNet1D(Bottleneck, [3, 4, 6, 3], **kwargs)