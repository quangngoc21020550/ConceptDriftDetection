import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import args

N_CLASSES = 4

class FineTunedNet(nn.Module):
    """
    normal feed-network
    """

    def __init__(self, centroid_matrix):
        super(FineTunedNet, self).__init__()
        self.fc1 = nn.Linear(args.centord_Vector_Length, N_CLASSES)
        self.fc1.weight = nn.Parameter(centroid_matrix)
        # self.fc2 = nn.Linear(250, 250)
        # self.fc3 = nn.Linear(250, 250)
        # self.fc4 = nn.Linear(250, args.centord_Vector_Length)

    def forward(self, x):
        x = F.softmax(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)
        return x

