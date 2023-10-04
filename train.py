import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR

from net import BITES
from utils import loss_shalit, loss_shaham


def fit(config, dataloader, p,type_loss='shalit'):
    net = BITES(in_features=config['input_dim'], num_nodes_shared=config["shared_layer"],
                num_nodes_indiv=config["individual_layer"], out_features=1,
                dropout=config["dropout"])
    optimizer = torch.optim.Adam(list(net.parameters()), lr=config["lr"], weight_decay=config["weight_decay"])
    # optimizer = SGLD(list(net.parameters()) , lr=config["lr"],momentum= 0.9)# ,weight_decay=config["weight_decay"])

    train_loss_list = []

    lr_scheduler = StepLR(optimizer, step_size=75, gamma=0.1)
    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        train_loss = 0.0

        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            x, treatments, y, _, _, _ = data
            optimizer.zero_grad()
            h0, h1, features = net(x)

            if type_loss == 'shalit':
                loss, loss_value = loss_shalit(h0, h1, features, y, treatments,p, alpha=config["alpha"],
                                               blur=config["blur"])

            elif type_loss == "shaham":
                loss, loss_value = loss_shaham(h0, h1, features, y, treatments,p)

            loss.backward()
            optimizer.step()
            train_loss += loss_value.item()
            epoch_steps += 1
        lr_scheduler.step()

        train_loss_list.append(train_loss / epoch_steps)

    print("Finished Training")
    return net, train_loss_list
