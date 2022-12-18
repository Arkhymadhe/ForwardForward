import torch


def collate_fn(batch, n_classes, corrupt=False):
    img, label = batch
    for i in range(img.shape[0]):
        labels = torch.zeros(1, n_classes)

        if corrupt:
            k = torch.randint(low=0, high=10, size=(1,)).item()
            while k == label[i]:
                k = torch.randint(low=0, high=10, size=(1,)).item()
        else:
            k = label[i]

        labels[:, k] = 1

        for ch in range(img.shape[1]):
            img[i, ch, :1, :n_classes] = labels.float().squeeze()

    return img