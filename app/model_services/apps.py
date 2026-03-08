import os
import sys
import json
import torch
from django.apps import AppConfig
from django.conf import settings
from torchvision.transforms import v2

class ModelServicesConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'model_services'
    
    model = None
    device = None
    transforms = None
    class_names = [] 
    
    def ready(self):
        PROJECT_ROOT = settings.BASE_DIR.parent
        
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.append(str(PROJECT_ROOT))
            
        json_path = os.path.join(settings.BASE_DIR, 'model_services', 'class_names.json')
        
        with open(json_path, "r", encoding="utf-8") as f:
            ModelServicesConfig.class_names = json.load(f)

        from conf.config import load_config
        from models.model_factory import ModelFactory
        
        config_path = os.path.join(PROJECT_ROOT, "conf", "config.yaml")
        config = load_config(config_path)
        
        ModelServicesConfig.device = torch.device('cpu')
        ModelServicesConfig.model = ModelFactory.create_model(config.active_model_config)
        
        checkpoint_path = os.path.join(
            PROJECT_ROOT, 
            config.training.checkpoint_dir, 
            f"{config.active_model_config.model_name}_best.pth"
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=ModelServicesConfig.device)
        ModelServicesConfig.model.load_state_dict(checkpoint["model_state_dict"])
        
        ModelServicesConfig.model.to(ModelServicesConfig.device)
        ModelServicesConfig.model.eval()
        
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        ModelServicesConfig.transforms = v2.Compose([
            v2.Resize(size=(256, 256), antialias=True),
            v2.CenterCrop(size=(config.dataset.image_size, config.dataset.image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])