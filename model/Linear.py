from torch import nn


class Linear(nn.Module):
    def __init__(self, i_f, o_f=1):
        super().__init__()
        self.linear = nn.Linear(i_f, o_f)

    def forward(self, x):
        x = self.linear(x)

        return x
