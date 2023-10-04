# networks
import torch
from torch import nn


class DenseVanillaBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=True, batch_norm=True, dropout=0., activation=nn.ELU,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        torch.manual_seed(0)
        self.linear = nn.Linear(in_features, out_features, bias)
        if w_init_:
            w_init_(self.linear.weight.data)
        self.activation = activation()
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, input):
        input = self.activation(self.linear(input))
        if self.batch_norm:
            input = self.batch_norm(input)
        if self.dropout:
            input = self.dropout(input)
        return input


class MLPVanilla(nn.Module):
    def __init__(self, in_features, num_nodes, out_features, batch_norm=True, dropout=None, activation=nn.ReLU,
                 output_activation=None, output_bias=True,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        torch.manual_seed(0)

        num_nodes = np.append(in_features, num_nodes)
        if not hasattr(dropout, '__iter__'):
            dropout = [dropout for _ in range(len(num_nodes) - 1)]
        net = []
        for n_in, n_out, p in zip(num_nodes[:-1], num_nodes[1:], dropout):
            net.append(DenseVanillaBlock(n_in, n_out, True, batch_norm, p, activation, w_init_))
        net.append(nn.Linear(num_nodes[-1], out_features, output_bias))
        if output_activation:
            net.append(output_activation)
        self.net = nn.Sequential(*net)

    def forward(self, input):
        return self.net(input)


# networks
import numpy as np
from torch import nn


class DenseVanillaBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=True, batch_norm=True, dropout=0., activation=nn.ELU,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        torch.manual_seed(0)
        self.linear = nn.Linear(in_features, out_features, bias)
        if w_init_:
            w_init_(self.linear.weight.data)
        self.activation = activation()
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, input):
        input = self.activation(self.linear(input))
        if self.batch_norm:
            input = self.batch_norm(input)
        if self.dropout:
            input = self.dropout(input)
        return input


class MLPVanilla1(nn.Module):
    def __init__(self, in_features, num_nodes, out_features, batch_norm=True, dropout=None, activation=nn.ReLU,
                 output_activation=None, output_bias=True,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        torch.manual_seed(0)

        num_nodes = np.append(in_features, num_nodes)
        if not hasattr(dropout, '__iter__'):
            dropout = [dropout for _ in range(len(num_nodes) - 1)]
        net = []
        for n_in, n_out, p in zip(num_nodes[:-1], num_nodes[1:], dropout):
            net.append(DenseVanillaBlock(n_in, n_out, True, batch_norm, p, activation, w_init_))
        net.append(nn.Linear(num_nodes[-1], out_features, output_bias))
        if output_activation:
            net.append(output_activation)
        self.net = nn.Sequential(*net)

    def forward(self, input):
        x = self.net[0](input)
        for layer in self.net[1:-1]:
            x = x + layer(x)

        x = self.net[-1](x)


        return x


class BITES(nn.Module):
    def __init__(self, in_features, num_nodes_shared, num_nodes_indiv, out_features,
                 num_treatments=2, batch_norm=True, dropout=None):
        super().__init__()

        self.shared_net = MLPVanilla1(
            in_features, num_nodes_shared[:-1], num_nodes_shared[-1],
            batch_norm, dropout,
        )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(2):
            net = MLPVanilla(
                num_nodes_shared[-1], num_nodes_indiv, out_features,
                batch_norm, dropout, output_bias=False)
            self.risk_nets.append(net)

    def forward(self, input):
        features = self.shared_net(input)

        h0 = self.risk_nets[0](features)
        h1 = self.risk_nets[1](features)

        return h0, h1, features

    def predict(self, x):
        self.eval()
        h0, h1, features = self(x)
        predictions = torch.stack([h0, h1], dim=1)

        self.train()
        return predictions

    def predict_numpy(self, x):
        self.eval()
        x = torch.Tensor(x)
        h0, h1, features = self(x)
        predictions = torch.stack([h0, h1], dim=1)
        self.train()

        return predictions.detach().numpy()
