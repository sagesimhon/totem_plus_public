import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
import torch.nn.functional as F


# Model
class MLP(nn.Module):
    def __init__(self, D=8, W=512, input_ch=2, output_ch=2, skips=None):
        """
        """
        super(MLP, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips or [4]

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D - 1)]
        )
        self.output_linear = nn.Linear(W, output_ch)

        # Initialize weights of the model
        # self.apply(utils.init_weights)

    def forward(self, x):
        input_pts = torch.split(x, [self.input_ch], dim=-1)[0]
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        output = self.output_linear(h)
        return output