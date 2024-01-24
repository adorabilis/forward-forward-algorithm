import matplotlib.pyplot as plt
import random
import torch

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Lambda


def load_mnist(train_batch_size=60_000, test_batch_size=10_0000):
    transform = Compose([ToTensor(), Lambda(lambda x: torch.flatten(x))])

    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def visualise_sample(X, name=""):
    X_ = X.reshape(28, 28)
    plt.figure(figsize=(4, 4))
    plt.title(name)
    plt.imshow(X_, cmap="gray")
    plt.show()


def overlay_y_on_x(X, y):
    num_samples = X.shape[0]
    X_ = X.clone()
    X_[:, :10] *= 0.0
    X_[range(0, num_samples), y] = X_.max()
    return X_


def create_pos_data(X, y):
    return overlay_y_on_x(X, y)


def create_neg_data(X, y):
    y_ = y.clone()
    for idx, label in enumerate(y_):
        labels = list(range(10))
        labels.remove(label)
        y_[idx] = random.choice(labels)
    return overlay_y_on_x(X, y_)


def demo_samples(X, y):
    visualise_sample(X[0], name="original")
    visualise_sample(create_pos_data(X[:1], y[:1]), name="positive")
    visualise_sample(create_neg_data(X[:1], y[:1]), name="negative")
