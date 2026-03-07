import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from .transforms import get_transforms

class Food101DataBuilder:
    def __init__(self, config: dict):
        self.cfg = config['dataset']
        self.data_dir = self.cfg['data_dir']
        
        self.train_transform = get_transforms(self.cfg['image_size'], is_train=True)
        self.val_transform = get_transforms(self.cfg['image_size'], is_train=False)
    
    def build_datasets(self):
        train_dataset = Food101(
            root=self.data_dir,
            split='train',
            download=self.cfg['download'],
            transform=self.train_transform
        )
        
        val_dataset = Food101(
            root=self.data_dir,
            split='test',
            download=self.cfg['download'],
            transform=self.val_transform
        )
        return train_dataset, val_dataset
    
    def get_dataloaders(self):
        
        train_ds, val_ds = self.build_datasets()
        
        train_loader = DataLoader(
            dataset=train_ds,
            batch_size=self.cfg['batch_size'],
            shuffle=True,
            num_workers=self.cfg['num_workers'],
            pin_memory=self.cfg['pin_memory'],
            drop_last=True
        )
        val_loader = DataLoader(
            dataset=val_ds,
            batch_size=self.cfg['batch_size'],
            shuffle=False,
            num_workers=self.cfg['num_workers'],
            pin_memory=self.cfg['pin_memory']
        )

        return train_loader, val_loader