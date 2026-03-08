import os
import torch
import sys
import json
from django.apps import AppConfig
from torchvision.transforms import v2

class WebappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'model_services'
    
    model = None
    device = None
    transforms = None
    class_names  = []
    
    def ready(self):
        if os.environ.get('RUN_MAIN', None) != 'true':
            return
        print("[INFO] Initialting model in memory...")
        
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'class_names.json')
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                WebappConfig.class_names = json.load(f)
                
            print(f"[SUCCESS] Class names loaded from {json_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"[FATAL] Class names file not found at {json_path}. Please run the script to extract class names.")
        
        from conf.config import load_config
        from models.model_factory import ModelFactory
        
        config = load_config("conf/config.yaml")
        
        self.device = torch.device('cpu')
        self.model = ModelFactory.create_model(config.active_model_config)
        
        checkpoint_path = os.path.join(config.training.checkpoint_dir, f"{config.active_model_config.model_name}_best.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"[FATAL] Model checkpoint not found at {checkpoint_path}. Please ensure the model is trained and the checkpoint exists.")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        self.model.to(self.device)
        self.model.eval()
        

        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        self.transforms = v2.Compose([
            v2.Resize(size=(256, 256), antialias=True),
            v2.CenterCrop(size=(config.dataset.image_size, config.dataset.image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        print("[SUCCESS] Model FoodVisionBig loaded into memory")