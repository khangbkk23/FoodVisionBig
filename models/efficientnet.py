import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from conf.config import EfficientNetConfig

class FoodVisionModel(nn.Module):
    def __init__(self, config: EfficientNetConfig):
        super().__init__()
        self.config = config


        weights = EfficientNet_B2_Weights.DEFAULT if config.pretrained else None
        self.backbone = efficientnet_b2(weights=weights)

        # Gradient Freezing
        if not config.fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

            total_blocks = len(self.backbone.features)
            
            # unfreeze N last block
            for i, block in enumerate(self.backbone.features):
                if i >= total_blocks - config.unfreeze_layers:
                    for param in block.parameters():
                        param.requires_grad = True
                        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, config.num_classes)
        )

    def forward(self, x):
        return self.backbone(x)