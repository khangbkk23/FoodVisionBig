import torch
import torch.nn as nn
from conf.config import CustomCNNConfig

class CustomFoodCNN(nn.Module):
    def __init__(self, config: CustomCNNConfig):
        super().__init__()
        self.config = config
        
        # Block 1
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(config.in_channels, config.base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(config.base_filters, config.base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(config.base_filters * 2, config.base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.base_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()

        fc_in_features = (config.base_filters * 4) * 7 * 7
        self.classifier = nn.Sequential(
            nn.Linear(fc_in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(512, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x