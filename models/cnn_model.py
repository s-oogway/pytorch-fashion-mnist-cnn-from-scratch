import torch
import torch.nn.functional as F
import torch.nn as nn


class CNNModel(nn.Module):
    """
    Rudimentary Convolutional Neural Network, coded from scratch, for the FashionMNIST dataset.

    Overall Architecture :
     - Two CNN layers each with ReLU activation & max pooling
     - Flattens feature maps
     - Two linear layers, ReLU after the first one
     - Outputs logits for 10 classes


    """
    def __init__(self):
        super().__init__()

        #CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)

        #Fully connected layers
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)

        self.fc2 = nn.Linear(in_features=128, out_features=10) #10 output classes for FashionMNIST

    def forward(self, x):
        # Input 'x' shape: [batch_size, 1, 28, 28]
        #Structure of conv blocks: Conv2d -> ReLU -> Maxpool
        #Apply first conv block
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        #Apply second conv block 
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = torch.flatten(x, 1)

        #Apply fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x