import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms as T
import os

from matplotlib import pyplot as plt

from model import Layer, Model
from utils import collate_fn

def main():

    dataset = MNIST(root=os.getcwd().replace('scripts', ''), train=True, download=False, transform=T.ToTensor())
    dl = DataLoader(dataset, 32, shuffle=True)
    args = {
        'in_channels': 1,
        'out_channels': 8,
        'kernel_size': 3,
        'bias': False
    }

    kwargs = {
        '0': {
            'base_layer': nn.Conv2d,
            'kwargs': {
                'in_channels': 1,
                'out_channels': 16,
                'kernel_size': 3,
                'bias': False
            }
        },
        '1': {
            'base_layer': nn.Conv2d,
            'kwargs': {
                'in_channels': 16,
                'out_channels': 8,
                'kernel_size': 3,
                'bias': False
            }
        },
        '2': {
            'base_layer': nn.Linear,
            'kwargs': {
                'in_features': 8 * 24 * 24,
                'out_features': 300,
            }
        },
        '3': {
            'base_layer': nn.Linear,
            'kwargs': {
                'in_features': 300,
                'out_features': 300,
            }
        },
        '4': {
            'base_layer': nn.Linear,
            'kwargs': {
                'in_features': 300,
                'out_features': 100,
            }
        },
        '5': {
            'base_layer': nn.Linear,
            'kwargs': {
                'in_features': 100,
                'out_features': 20,
            }
        }
    }
    model = Model(1e-3, 2, **kwargs, device = 'cuda', num_classes=10, epochs=10)
    layer = Layer(nn.Conv2d, 1e-4, 2, 10, **args, device='cuda')

    pos_data = torch.randn((32, 1, 28, 28), dtype=torch.float32)
    neg_data = torch.randn((32, 1, 28, 28), dtype=torch.float32)

    a, b = layer.train_layer(pos_data, neg_data)

    # print(a.shape)
    # print("Argmax: ", model.infer(neg_data))
    print("All is well\n")

    epochs = 10
    n_classes = 10

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch} started...\n")
        accs = list()
        for batch in dl:
            pos_data = collate_fn(batch, n_classes, False)
            neg_data = collate_fn(batch, n_classes, True)

            model.train_model(pos_data, neg_data)

            pred_labels = model(batch[0])
            accs.append(100 * sum(pred_labels == batch[1].to(model.device)) / pred_labels.shape[0])

        print(f"  Epoch [{epoch}/{epochs}]; Accuracy (%): {sum(accs) / len(accs) : .4f}\n")


if __name__ == '__main__':
    main()
