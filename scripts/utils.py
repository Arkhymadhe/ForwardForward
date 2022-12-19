import torch


def collate_fn(batch, n_classes, corrupt=False):
    img, label = batch

    labels = torch.zeros(len(label), n_classes)

    if corrupt:
        labels_ = torch.randint(low=0, high=10, size=(len(label),))
        labels[range(len(label)), labels_] = 1
    else:
        labels[range(len(label)), label] = 1

    for ch in range(img.shape[1]):
        img[:, ch, 0, :n_classes] = labels.float()

    return img


