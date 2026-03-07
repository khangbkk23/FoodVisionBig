import torch
from torchvision.transforms import v2

def get_transforms(image_size: int, is_train: bool):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    if is_train:
        return v2.Compose([
            v2.RandomResizedCrop(size=(image_size, image_size), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return v2.Compose([
            v2.Resize(size=(256, 256), antialias=True),
            v2.CenterCrop(size=(image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])