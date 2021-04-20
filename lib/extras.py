
import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features,
                 sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(
            in_features, out_features, bias=bias)

        # create a matrix for sigma
        w = torch.full((out_features, in_features), sigma_init)
 
        # to make the sigmas trainable, wrap the tensor in nn.Parameter
        self.sigma_weight = nn.Parameter(w)

        # register buffer is used to register a buffer that 
        # should not be considered a model parameter. B 
        # Buffers are named tensors that do not update 
        # gradients at every step, like parameters. 
        z = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", z)

        # an extra bias and buffer are created for
        # bias of the layer
        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)
            z = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", z)

        # overriden from nn.lineae
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialisation of nn.linear weight and bias 
        """
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        """
        sample random noise in both the weights and bias 
        buffers, and perform linear transformation of the 
        input data in the same way as nn.linear does
        """
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * \
                   self.epsilon_bias.data
        v = self.sigma_weight * self.epsilon_weight.data + \
            self.weight
        return F.linear(input, v, bias)