import torch
from torch import nn, optim
import matplotlib.pyplot as plt

def loss(data, labels):
    return (data - labels).mean()

def new_loss(p, n):
    return (p - n).mean()

class Layer(nn.Module):
    def __init__(self, base_layer, lr, threshold, epochs, device, **kwargs):
        super(Layer, self).__init__()
        self.layer = base_layer(**kwargs)

        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.threshold = threshold
        self.act_layer = nn.ReLU()
        self.layer = self.layer.to(self.device)

        self.opt = optim.Adam(self.layer.parameters(), lr=self.lr, betas=(0.9, 0.999))

    def train_layer(self, pos_data, neg_data):

        pos_data = pos_data.to(self.device)
        neg_data = neg_data.to(self.device)

        for e in range(self.epochs):
            pos_act = self.data_pass(pos_data)
            #pos_loss = -self.calc_loss(pos_act)
            #pos_loss.backward()

            neg_act = self.data_pass(neg_data)
            #neg_loss = self.calc_loss(neg_act)
            #neg_loss.backward()

            #loss = -new_loss(pos_act, neg_act)
            loss = torch.log(1 + torch.exp(torch.cat([
                -pos_act + self.threshold,
                neg_act - self.threshold]))).mean()

            loss.backward()

            self.opt.step()
            self.opt.zero_grad()

        print(f"Layer {self.layer._get_name()} trained!\n")

        return self.act_layer(self.layer(pos_data)).detach(), self.act_layer(self.layer(neg_data)).detach()

    def data_pass(self, data):
        act = self.act_layer(self.layer(data.to(self.device)))
        return act.pow(2).mean(1)

    def train(self, pos_data, neg_data):
        print("Generate positive data...\n")
        pos_data = pos_data.to(self.device)

        print("Generate negative data...\n")
        neg_data = neg_data.to(self.device)

        for e in range(self.epochs):
            _, _ = self.train_layer(pos_data, neg_data)

        print(f"Layer {self.layer._get_name()} trained!\n")
        return self.train_layer(pos_data, neg_data)

    def calc_loss(self, data):
        labels = data.clone()
        labels.fill_(self.threshold)
        return loss(data, labels)

    def forward(self, x):
        return self.act_layer(self.layer(x.to(self.device)))


class Model(nn.Module):
    def __init__(self, lr, threshold, epochs, device, num_classes = 2, **kwargs):
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.num_layers = len(kwargs)

        self.layers = nn.Sequential()

        for i in range(self.num_layers):
            self.layers.add_module(
                name = f'layer_{i}',
                module = Layer(
                    kwargs[f'{i}']['base_layer'],
                    self.lr,
                    self.threshold,
                    self.epochs,
                    **kwargs[f"{i}"]['kwargs'],
                    device=self.device
                )
            )

        self.layers = self.layers.to(self.device)

    def train_model(self, pos_data, neg_data):

        pos_data = pos_data.to(self.device)
        neg_data = neg_data.to(self.device)

        for layer in self.layers:
            if layer.layer.__class__.__name__ == 'Linear':
                pos_data, neg_data = pos_data.view(pos_data.shape[0], -1), neg_data.view(neg_data.shape[0], -1)

            pos_data, neg_data = layer.train_layer(pos_data, neg_data)

        return

    def forward(self, data):
        overall_fit_metric = list()
        fit_metric = list()

        for class_ in range(self.num_classes):
            fit_metric.clear()
            pred = data.to(self.device)

            for layer in self.layers:
                if layer.layer.__class__.__name__ == 'Linear':
                    pred = pred.view(pred.shape[0], -1)
                pred = layer(pred)
                fit_metric.append(pred.view(pred.shape[0], -1).pow(2).mean(1).unsqueeze(1))

            #fit = torch.sum(torch.tensor(fit_metric), keepdim=True)
            #print("Shape is: ",fit_metric[0].shape)
            sum_ = 0
            for t in fit_metric:
                sum_ += t

            overall_fit_metric.append(sum_)

        over_all = torch.cat(overall_fit_metric, dim=1)
        over_all_sum = torch.sum(over_all, dim=-1, keepdim=True)
        over_all /= over_all_sum
        #print("Overall shape: ", over_all.shape)
        #print(over_all[0])
        #over_all = torch.tensor(overall_fit_metric)/sum(overall_fit_metric)

        return torch.argmax(over_all, dim=-1)



