# File: model_cnn.py
import torch.nn as nn
import torchvision.models as models

class ChartCNN(nn.Module):
    def __init__(self, num_classes=15):
        super(ChartCNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

# Cách sử dụng:
# model = ChartCNN(num_classes=15)
# model.to(device)
