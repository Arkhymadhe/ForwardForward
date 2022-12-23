import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms as T

import os

from matplotlib import pyplot as plt

from model import NetLayer, Model
from utils import collate_fn


def main():
    batch_size = 32

    print("Get train data...\n")
    train_dataset = MNIST(
        root=os.getcwd().replace('scripts', ''),
        train=True, download=True,
        transform=T.Compose(
            [T.ToTensor(), T.Normalize((0.5,), (0.5,))]
        )
    )

    print("Get test data...\n")
    test_dataset = MNIST(
        root=os.getcwd().replace('scripts', ''),
        train=False, download=True,
        transform=T.Compose(
            [T.ToTensor(), T.Normalize((0.5,), (0.5,))]
        )
    )

    train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
                'out_channels': 5,
                'kernel_size': 3,
                'bias': False
            }
        },
        '1': {
            'base_layer': nn.Conv2d,
            'kwargs': {
                'in_channels': 5,
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
                'out_features': 100,
            }
        },
        '4': {
            'base_layer': nn.Linear,
            'kwargs': {
                'in_features': 100,
                'out_features': 20,
            }
        }
    }

    model = Model(1e-3, 2, **kwargs, device = device, num_classes=10, epochs=1000)
    layer = NetLayer(nn.Conv2d, 1e-4, 2, 10, **args, device=device)

    pos_data = torch.randn((32, 1, 28, 28), dtype=torch.float32)
    neg_data = torch.randn((32, 1, 28, 28), dtype=torch.float32)

    a, b = layer.train_layer(pos_data, neg_data)

    # print(a.shape)
    # print("Argmax: ", model.infer(neg_data))
    print("All is well\n")

    n_classes = 10

    accs = list()
    for i, batch in enumerate(train_dl):
        pos_data = collate_fn(batch, n_classes, False)
        neg_data = collate_fn(batch, n_classes, True)

        model.train_model(pos_data, neg_data)

        pred_labels = model(batch[0])
        accs.append(100 * pred_labels.eq(batch[1].to(device)).float().mean())

    print(f"Train Accuracy (%): {sum(accs) / len(accs) : .4f}\n")
    accs.clear()

    for i, batch in enumerate(test_dl):
        pred_labels = model(batch[0])
        accs.append(100 * pred_labels.eq(batch[1].to(device)).float().mean())

    print(f"Test Accuracy (%): {sum(accs) / len(accs) : .4f}\n")


if __name__ == '__main__':
    main()
