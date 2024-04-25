import numpy as np
import torch

from pathlib import Path
from PIL import Image
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split


def save_images(images, labels, directory):
    for i, (image, label) in enumerate(zip(images, labels)):
        folder = directory / str(label)
        folder.mkdir(exist_ok=True)

        image_path = folder / f"{i}.png"
        Image.fromarray(image.astype(np.uint8)).save(image_path)


def get_dataset(loader):
    for images, labels in loader:
        return images.squeeze().numpy() * 255, labels.numpy()


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    dataset = MNIST(
        root='.',
        download=True,
        transform=ToTensor()
    )

    length = len(dataset)
    train_size = int(0.80 * length)
    val_size = int(0.10 * length)
    test_size = length - (train_size + val_size)

    train, validation, test = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train,
        batch_size=len(train),
        shuffle=True
    )

    val_loader = DataLoader(
        validation,
        batch_size=len(validation),
        shuffle=False
    )

    test_loader = DataLoader(
        test,
        batch_size=len(test),
        shuffle=False
    )

    train_images, train_labels = get_dataset(train_loader)
    val_images, val_labels = get_dataset(val_loader)
    test_images, test_labels = get_dataset(test_loader)

    mnist_dir = Path('mnist_dataset')
    mnist_dir.mkdir(parents=True, exist_ok=True)

    train_dir = mnist_dir / 'training'
    train_dir.mkdir(exist_ok=True)

    val_dir = mnist_dir / 'validation'
    val_dir.mkdir(exist_ok=True)

    test_dir = mnist_dir / 'testing'
    test_dir.mkdir(exist_ok=True)

    save_images(
        train_images,
        train_labels,
        train_dir
    )

    save_images(
        val_images,
        val_labels,
        val_dir
    )

    save_images(
        test_images,
        test_labels,
        test_dir
    )

    print(f"MNIST images saved in '{mnist_dir}' directory.")


if __name__ == '__main__':
    main()
