import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(4, 4), stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)

        ## Max-Pool layer
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        ## Linear layers
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=136)

        ## Dropout
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout6 = nn.Dropout(p=0.6)

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(num_features=96, eps=1e-05)
        self.bn2 = nn.BatchNorm2d(num_features=256, eps=1e-05)
        self.bn3 = nn.BatchNorm2d(num_features=384, eps=1e-05)
        self.bn4 = nn.BatchNorm2d(num_features=384, eps=1e-05)
        self.bn5 = nn.BatchNorm2d(num_features=256, eps=1e-05)
        self.bn6 = nn.BatchNorm1d(num_features=4096, eps=1e-05)
        self.bn7 = nn.BatchNorm1d(num_features=4096, eps=1e-05)


    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout2(x)

        x = F.elu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)

        x = F.elu(self.conv3(x))
        x = self.bn3(x)
        x = self.dropout4(x)

        x = F.elu(self.conv4(x))
        x = self.bn4(x)
        x = self.dropout4(x)

        x = F.elu(self.conv5(x))
        x = self.bn5(x)
        x = self.pool(x)

        ## Flatten
        x = x.view(x.size(0), -1)

        ## Fully connected layers
        x = F.elu(self.fc1(x))
        x = self.bn6(x)
        x = self.dropout6(x)

        x = F.elu(self.fc2(x))
        x = self.bn6(x)
        x = self.dropout6(x)

        x = self.fc3(x)

        return x
