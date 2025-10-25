import torch.nn as nn
import torch.nn.functional as F

class CNN_EMNIST(nn.Module):
    def __init__(self, class_num):
        super(CNN_EMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, class_num)
        
    def forward(self, x, return_feat=False):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = F.relu(self.conv3(x))             # 7x7 -> 7x7
        x = self.pool(x)                      # 7x7 -> 3x3
        x = self.dropout1(x)
        
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        feat = x
        x = self.fc2(x)
        
        if return_feat:
            return x, feat
        return x

def cnn_emnist_emnist(args):
    return CNN_EMNIST(class_num=args.class_num)