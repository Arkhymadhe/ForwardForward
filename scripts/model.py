import torch
from torch import nn, optim
import matplotlib.pyplot as plt

from itertools import chain
from utils import embed_data


def loss(data, labels):
    return (data - labels).mean()


def new_loss(p, n):
    return (p - n).mean()


class NetLayer(nn.Module):
    def __init__(self, base_layer, lr, threshold, epochs, device, **kwargs):
        super(NetLayer, self).__init__()
        #self.layer = base_layer(**kwargs)

        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.threshold = threshold
        #self.act_layer = nn.LeakyReLU(negative_slope=.2, inplace=True)

        self.layer = nn.Sequential(
            base_layer(**kwargs),
            nn.LeakyReLU(negative_slope=.2, inplace=False)
        )
        self.layer = self.layer.to(self.device)

        self.opt = optim.Adam(
            self.layer.parameters(),
            lr=self.lr, betas=(0.9, 0.99)
        )

    def train_layer(self, pos_data, neg_data):

        h_pos = pos_data
        h_neg = neg_data

        if self.layer[0].__class__.__name__ == 'Linear':
            h_pos, h_neg = h_pos.view(h_pos.shape[0], -1), h_neg.view(h_neg.shape[0], -1)

        for e in range(1, self.epochs + 1):
            self.opt.zero_grad()

            pos_act = self.forward(h_pos).pow(2).mean(1)
            # pos_loss = -self.calc_loss(pos_act)
            # pos_loss.backward()

            neg_act = self.forward(h_neg).pow(2).mean(1)
            # neg_loss = self.calc_loss(neg_act)
            # neg_loss.backward()

            # loss = -new_loss(pos_act, neg_act)
            loss = torch.log(1 + torch.exp(torch.cat([
                -pos_act + self.threshold,
                neg_act - self.threshold]))).mean()

            loss.backward()

            self.opt.step()

            self.opt.zero_grad()

        return self.forward(h_pos).detach(), self.forward(h_neg).detach()

    def data_pass(self, data):
        act = self.layer(data)
        return act.pow(2).mean(1)

    def train(self, pos_data, neg_data):
        print("Generate positive data...\n")
        h_pos = pos_data

        print("Generate negative data...\n")
        h_neg = neg_data

        for e in range(self.epochs):
            _, _ = self.train_layer(h_pos, h_neg)

        return self.train_layer(h_pos, h_neg)

    def calc_loss(self, data):
        labels = data.clone()
        labels.fill_(self.threshold)
        return loss(data, labels)

    def forward(self, x):
        x = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        if self.layer[0].__class__.__name__ == 'Linear':
            x = x.view(x.shape[0], -1)
        return self.layer(x)


class Model(nn.Module):
    def __init__(self, lr=1e-6, threshold=1., epochs=10, device='cuda', num_classes=2, **kwargs):
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.num_layers = len(kwargs)

        self.layers = nn.Sequential().to(self.device)

        for i in range(self.num_layers):
            self.layers.add_module(
                name=f'layer_{i}',
                module=NetLayer(
                    kwargs[f'{i}']['base_layer'],
                    self.lr,
                    self.threshold,
                    self.epochs,
                    **kwargs[f"{i}"]['kwargs'],
                    device=self.device
                )
            )

        #self.layers = self.layers.to(self.device)

    def train_model(self, pos_data, neg_data):

        h_pos = pos_data
        h_neg = neg_data

        for layer in self.layers:
            h_pos, h_neg = layer.train_layer(h_pos, h_neg)

        return

    def forward(self, data):
        overall_fit_metric = list()
        fit_metric = list()

        for class_ in range(self.num_classes):
            fit_metric.clear()
            pred = embed_data(data, class_, self.num_classes, False)
            #pred = data

            for layer in self.layers:
                pred = layer(pred)
                fit_metric.append(pred.view(pred.shape[0], -1).pow(2).mean(1))

            overall_fit_metric.append(sum(fit_metric).unsqueeze(1))

        over_all = torch.cat(overall_fit_metric, dim=1)
        over_all_sum = torch.sum(over_all, dim=-1, keepdim=True)

        over_all /= over_all_sum

        print(over_all)

        return torch.argmax(over_all, dim=-1)
