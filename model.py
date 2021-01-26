import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Network 만들기
"""

# Input -> Convolution -> C1 -> Subsampling -> S2 -> Convolutions -> C3 -> Subsampling -> S4 
# -> Full connection -> C5 -> Full connectino -> F6 -> Gaussian connection -> Output

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
    
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.c5 = nn.Linear(16*4*4, 120)
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)
 
    def forward(self, x):
        x = self.c1(x)
        x = F.max_pool2d(x, kernel_size=2) # S2포함
        x = self.c3(x)
        x = F.max_pool2d(x, kernel_size=2) # S4포함
        x = x.view(-1, 16*4*4)
        x = self.c5(x)
        x = self.f6(x)
        x = self.output(x)
        return x


# model = LeNet().to(device)

# print(summary(model, input_size=(1,28,28), batch_size=batch_size, device=device))
