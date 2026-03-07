# models/model_factory.py
import torch.nn as nn
from typing import Union
from conf.config import CustomCNNConfig, EfficientNetConfig
from models.efficientnet import FoodVisionModel

from models.custom_cnn import CustomFoodCNN 

class ModelFactory:
    @staticmethod
    def create_model(config: Union[CustomCNNConfig, EfficientNetConfig]) -> nn.Module:
        if isinstance(config, EfficientNetConfig):
            print(f"[INFO] Initiating benchmark model: {config.model_name}")
            return FoodVisionModel(config)
            
        elif isinstance(config, CustomCNNConfig):
            print(f"[INFO] Initiating my model: {config.model_name}")
            return CustomFoodCNN(config)
            
        else:
            raise TypeError(f"[FATAL] Loại cấu hình {type(config)} không được hệ thống hỗ trợ!")