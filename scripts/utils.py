import torch


def collate_fn(batch, n_classes, corrupt=False):
    imgs, label = batch

    img = imgs.clone()

    labels = torch.zeros(len(label), n_classes)

    img_max = img.max().item()

    if corrupt:
        labels_ = torch.randint(low=0, high=n_classes, size=(len(label),))
        labels[range(len(label)), labels_] = img_max
    else:
        labels[range(len(label)), label] = img_max

    for ch in range(img.shape[1]):
        img[:, ch, 0, :n_classes] = labels.float()

    return img


def embed_data(data, metadata_label, n_classes=10, corrupt=False):
    batch_size = len(data)
    labels = torch.ones(batch_size,)
    labels.fill_(metadata_label)

    return collate_fn((data, labels.long()), n_classes, corrupt)

