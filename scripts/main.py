import shutil
import os

import torch

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms as T

from model import FFModel
from utils import collate_fn
from config import get_network_config


def main():
    batch_size = 128*2

    print("Get train data...")
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
    test_dl = DataLoader(test_dataset, batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(device)

    kwargs = get_network_config('Conv')

    num_classes = 10
    #lr = .0015
    max_lr = 2e-1
    min_lr = 1.5e-1
    lr = 3e-5
    epochs = 100
    threshold = 2.

    model = FFModel(
        lr=lr, threshold=threshold,
        **kwargs,
        device=device,
        num_classes=num_classes,
        epochs=epochs
    )

    accs = list()

    # Train network
    print(f"Training with {len(train_dl)} batches...")
    for i, batch in enumerate(train_dl, start=1):
        pos_data = collate_fn(batch, num_classes, False).to(device)
        neg_data = collate_fn(batch, num_classes, True).to(device)

        if (i == 1) or (i % 50 == 0):
            print(f"  Training with batch {i}...")

        model.train_model(pos_data, neg_data)

    # Freeze network weights
    for layer in model.layers:
        layer.requires_grad_(False)

    # Training performance
    for i, batch in enumerate(train_dl, start=1):

        pred_labels = model(batch[0].to(device))

        accs.append(100 * pred_labels.eq(batch[1].to(device)).float().mean())

        print("Pred | Actual")
        for (a, b) in zip(pred_labels, batch[1].to(device)):
            print(f"{a} | {b}")

    print(f"\nTrain Accuracy (%): {sum(accs) / len(accs) : .4f}\n")
    accs.clear()

    # Test performance
    print(f"\nTesting with {len(test_dl)} batches...")
    for i, batch in enumerate(test_dl, start=1):
        if (i == 1) or (i % 10 == 0):
            print(f"  Testing with batch {i}...")

        pred_labels = model(batch[0].to(device))

        accs.append(100 * pred_labels.eq(batch[1].to(device)).float().mean().item())

    print(f"\nTest Accuracy (%): {sum(accs) / len(accs) : .4f}\n")

    model_name = 'model.ckpt'

    torch.save(
        {
            'model_state_dict': model.state_dict()
        },
        model_name
    )

    shutil.move(
        os.path.join(os.getcwd(), model_name),
        os.path.join(
            os.getcwd().replace('scripts', 'artefacts'),
            model_name
        )
    )


if __name__ == '__main__':
    main()
