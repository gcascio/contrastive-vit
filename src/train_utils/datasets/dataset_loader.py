import os
import torchvision.transforms as transforms
from torchvision import datasets
from train_utils.augmentation.simclr_transform import SimCLRTransform
from .image_folder import ImageFolder

from config.config import TrainConfig


def get_cifar100_dataset(config: TrainConfig, _: int):
    norm = transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    )

    train_dataset = datasets.CIFAR100(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            norm,
        ]),
    )

    test_dataset = datasets.CIFAR100(
        root=config.data_dir,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            norm,
        ]),
    )

    return train_dataset, test_dataset


def get_cifar10_dataset(config: TrainConfig, _: int):
    norm = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    )

    train_dataset = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            norm,
        ]),
    )

    test_dataset = datasets.CIFAR10(
        root=config.data_dir,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            norm,
        ]),
    )

    return train_dataset, test_dataset


def get_mnist_dataset(config: TrainConfig, _: int):
    norm = transforms.Normalize((0.1307,), (0.3081,))

    train_dataset = datasets.MNIST(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            norm,
        ]),
    )

    test_dataset = datasets.MNIST(
        root=config.data_dir,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            norm,
        ]),
    )

    return train_dataset, test_dataset


def get_imagenet_dataset(config: TrainConfig, image_size: int):
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_dataset = ImageFolder(
        os.path.join(config.data_dir, 'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ]),
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        norm,
    ])

    if config.resize_image_size:
        transform = transforms.Compose([
            transforms.Resize(config.resize_image_size),
            transforms.CenterCrop(image_size),
            transform,
        ])

    test_dataset = ImageFolder(
        os.path.join(config.data_dir, 'val'),
        transform,
    )

    return train_dataset, test_dataset


def get_augmented_imagenet_dataset(config: TrainConfig, image_size: int):
    simclr_transform = SimCLRTransform(image_size)

    train_dataset = ImageFolder(
        os.path.join(config.data_dir, 'train'),
        transforms.Compose([
            simclr_transform,
        ]),
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if config.resize_image_size:
        transform = transforms.Compose([
            transforms.Resize(config.resize_image_size),
            transforms.CenterCrop(image_size),
            transform,
        ])

    test_dataset = ImageFolder(
        os.path.join(config.data_dir, 'val'),
        transform,
    )

    return train_dataset, test_dataset


def get_fake_dataset(config: TrainConfig, image_size: int):
    default_dataset_size = 1200000
    default_test_dataset_size = 5000

    train_dataset = datasets.fakedata(
        size=config.dataset_size or default_dataset_size,
        image_size=(config.channels, image_size, image_size),
        num_classes=config.num_classes,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    test_dataset = datasets.fakedata(
        size=config.test_dataset_size or default_test_dataset_size,
        image_size=(config.channels, image_size, image_size),
        num_classes=config.num_classes,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    return train_dataset, test_dataset


def get_dataset(config: TrainConfig, image_size: int):
    init_dataset = ({
        'cifar10': get_cifar10_dataset,
        'cifar100': get_cifar100_dataset,
        'mnist': get_mnist_dataset,
        'imagenet': get_imagenet_dataset,
        'imagenet_contrastive': get_augmented_imagenet_dataset,
        'fake': get_fake_dataset,
    })[config.dataset]

    dataset = init_dataset(config, image_size)
    return dataset
