import torch
import torch.nn as nn
import torch.nn.functional as F


class DFCNN_CTC(nn.Module):
    def __init__(self, num_classes, freq_bins=200):
        super(DFCNN_CTC, self).__init__()

        # block1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # block2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # block3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv33 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # block4，不池化
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv44 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # block5，不池化
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv55 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 频率维经过 3 次池化：200 -> 100 -> 50 -> 25
        reduced_freq = freq_bins // 8
        self.fc1 = nn.Linear(128 * reduced_freq, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: [B, 1, T, F]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1(self.conv11(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn2(self.conv22(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn3(self.conv33(x)))
        x = self.pool(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn4(self.conv44(x)))

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn5(self.conv55(x)))

        # [B, C, T', F'] -> [B, T', C, F']
        x = x.permute(0, 2, 1, 3)

        # [B, T', C, F'] -> [B, T', C * F']
        x = x.contiguous().view(x.size(0), x.size(1), -1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # [B, T', num_classes]
        x = self.fc2(x)

        # CTCLoss 需要 log_softmax
        x = F.log_softmax(x, dim=-1)

        return x