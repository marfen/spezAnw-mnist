from torch.nn import Linear, Module, Conv2d, Dropout2d
import torch.nn.functional as F
import torch.nn as nn
import torch as t


class Model1(Module):
    def __init__(self, input_shape):
        super(Model1, self).__init__()

        self.conv1 = Conv2d(1, 10, 5)
        self.conv2 = Conv2d(10, 20, 5)
        self.fc1 = Linear(320, 80)         # 20(conv2 output) Bilder aufgeteilt in 4x4  = 320
        self.fc2 = Linear(80, 10)          # 10 outputs für 0-9



    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 320)                #ignoriert 1. Dimension und legt 320 pixel nebeneinander
        x = t.sigmoid(self.fc1(x))
        x = t.sigmoid(self.fc2(x))
        return F.log_softmax(x, dim=1)             # höchster output 1, rest 0 (klassifizierung)
