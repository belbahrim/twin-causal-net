import torch
from torch import nn
import torch.nn.functional as functional
from twincausal.proximal.proximal import LinearProximal


# 1-hidden layer twin-causal-model
class Smite1(nn.Module):
    def __init__(self, input_size, negative_slope=0.05, nb_neurons=1024, prune=False):
        super(Smite1, self).__init__()

        self.negative_slope = negative_slope
        self.nb_neurons = nb_neurons

        linear_cls = nn.Linear if not prune else LinearProximal

        self.fc_layer = linear_cls(input_size, nb_neurons)
        self.scaling_a = nn.Parameter(torch.randn(nb_neurons).abs_())
        self.scaling_b = nn.Parameter(torch.randn(nb_neurons).abs_())
        self.fc_output = linear_cls(nb_neurons, 1)

    def forward(self, x_treat, x_control, treat):

        x1 = self.fc_layer(x_treat)
        x1 = x1.mul(self.scaling_a - self.scaling_b)
        x1 = functional.leaky_relu(x1, self.negative_slope)
        # x1 = torch.sigmoid(x1)
        x1 = self.fc_output(x1)

        x0 = self.fc_layer(x_control)
        x0 = x0.mul(self.scaling_a - self.scaling_b)
        x0 = functional.leaky_relu(x0, self.negative_slope)
        # x0 = torch.sigmoid(x0)
        x0 = self.fc_output(x0)

        m1t = x1 * treat + x0 * (1.0 - treat)
        m1t = torch.sigmoid(m1t)

        m11 = torch.sigmoid(x1)
        m10 = torch.sigmoid(x0)

        return m1t, m11, m10


# 2 hidden layers SMITE
class Smite2(nn.Module):
    def __init__(self, input_size, negative_slope=0.05, nb_neurons=1024, prune=False):
        super(Smite2, self).__init__()

        self.negative_slope = negative_slope
        self.nb_neurons_per_layer = int(nb_neurons / 2)  # Make sure this is an integer

        linear_cls = nn.Linear if not prune else LinearProximal

        self.fc_layer1 = linear_cls(input_size, self.nb_neurons_per_layer)
        self.fc_layer2 = linear_cls(self.nb_neurons_per_layer, self.nb_neurons_per_layer)
        self.fc_output = linear_cls(self.nb_neurons_per_layer, 1)

    def forward(self, x_treat, x_control, treat):

        x1 = self.fc_layer1(x_treat)
        x1 = functional.leaky_relu(x1, self.negative_slope)
        # x1 = torch.sigmoid(x1)
        x1 = self.fc_layer2(x1)
        x1 = functional.leaky_relu(x1, self.negative_slope)
        x1 = self.fc_output(x1)

        x0 = self.fc_layer1(x_control)
        x0 = functional.leaky_relu(x0, self.negative_slope)
        # x0 = torch.sigmoid(x0)
        x0 = self.fc_layer2(x0)
        x0 = functional.leaky_relu(x0, self.negative_slope)
        x0 = self.fc_output(x0)

        x_return = x1 * treat + x0 * (1.0 - treat)
        x_return = torch.sigmoid(x_return)

        p1 = torch.sigmoid(x1)
        p0 = torch.sigmoid(x0)

        return x_return, p1, p0


#Smite 2 xxxx : Twin Causal ok

# Identifiable parameters, i.e. logistic regression SMITE

## Include them

### Structured & Unstructured pruning
#alpha1: for struc. & alpha2 for unstruc.
#default: alpha 2 will be l2 penalty
#default: same activation as here but give option to the users
# hidden layer




























class LogisticSmite(nn.Module):
    def __init__(self, input_size, prune=False):
        super(LogisticSmite, self).__init__()

        linear_cls = nn.Linear if not prune else LinearProximal

        self.fc_output = linear_cls(input_size, 1)

    def forward(self, x_treat, x_control, treat):

        x1 = self.fc_output(x_treat)
        x0 = self.fc_output(x_control)

        x_return = x1 * treat + x0 * (1.0 - treat)
        x_return = torch.sigmoid(x_return)

        p1 = torch.sigmoid(x1)
        p0 = torch.sigmoid(x0)

        return x_return, p1, p0





