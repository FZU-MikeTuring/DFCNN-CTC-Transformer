import torch
import torch.nn as nn
import torch.nn.functional as F


class DFCNN_CTC(nn.Module):
    def __init__(self, num_classes, freq_bins=200):
        super(DFCNN_CTC, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv11 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv22 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn22 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv33 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn33 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv44 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn44 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv55 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn55 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        reduced_freq = freq_bins // 8
        self.fc1 = nn.Linear(128 * reduced_freq, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn22(self.conv22(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn33(self.conv33(x)))
        x = self.pool(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn44(self.conv44(x)))

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn55(self.conv55(x)))

        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(x.size(0), x.size(1), -1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)
