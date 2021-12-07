import torch
import torchvision.transforms as transforms


class SimCLRTransform((torch.nn.Module)):
    """Algorithm from: https://arxiv.org/pdf/2002.05709.pdf
    (accessed 15.11.2021) translated to pytorch
    """

    def __init__(self, image_size):
        color_jitter = transforms.ColorJitter(
            0.8,  # brightness
            0.8,  # contrast
            0.8,  # saturation
            0.2,  # hue
        )

        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)
