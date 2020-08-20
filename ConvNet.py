import torch
import torch.nn.functional as F
from torch import nn

class ConvNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(64*13*13, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
#         self.classifier =  nn.Sequential(
#             nn.Dropout(0.001),
#             nn.Linear(64 * 13 * 13, 4096),
#             nn.ReLU(),
#             nn.Dropout(0.001),
#             nn.Linear(4096, 512),
#             nn.ReLU())
#         self.final_l = nn.Linear(512, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         out = self.final_l(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out