import torch
from torch import nn
import torch.nn.functional as functional
from twincausal.proximal.proximal import LinearProximal


# 1-hidden layer twin-causal-model
class TwinCausalModel(nn.Module):
    def __init__(self, input_size, negative_slope=0.05, nb_neurons=1024, prune=True):
        super(TwinCausalModel, self).__init__()

        self.negative_slope = negative_slope
        self.nb_neurons = nb_neurons
        self.prune = prune
        linear_cls = nn.Linear if not prune else LinearProximal

        self.fc_layer = linear_cls(input_size, nb_neurons)
        if prune:
            self.scaling_a = nn.Parameter(torch.randn(nb_neurons).abs_())
            self.scaling_b = nn.Parameter(torch.randn(nb_neurons).abs_())
        self.fc_output = linear_cls(nb_neurons, 1)

    def forward(self, x_treat, x_control, treat):

        x1 = self.fc_layer(x_treat)
        if self.prune:
            x1 = x1.mul(self.scaling_a - self.scaling_b)
        x1 = functional.leaky_relu(x1, self.negative_slope)
        x1 = self.fc_output(x1)
        m11 = torch.sigmoid(x1)

        x0 = self.fc_layer(x_control)
        if self.prune:
            x0 = x0.mul(self.scaling_a - self.scaling_b)
        x0 = functional.leaky_relu(x0, self.negative_slope)
        x0 = self.fc_output(x0)
        m10 = torch.sigmoid(x0)

        m1t = m11 * treat + m10 * (1.0 - treat)

        return m1t, m11, m10

