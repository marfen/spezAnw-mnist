from torch.nn import Linear, Module
import torch.nn.functional as F


class SimpleFC(Module):
    def __init__(self, input_shape):
        super(SimpleFC, self).__init__()

        # split input according to your network's design,
        # in this case a single incoming image split into its height, width and depth
        self.depth = input_shape[0]
        self.height = input_shape[1]
        self.width = input_shape[2]

        # network layers
        self.layer_1 = Linear(in_features=self.height * self.width * self.depth, out_features=200)
        self.layer_2 = Linear(in_features=200, out_features=200)
        self.layer_3 = Linear(in_features=200, out_features=10)

    def forward(self, x):
        x = x.view(-1, self.height * self.width)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return F.log_softmax(x, dim=1)
