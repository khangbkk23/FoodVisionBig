import os
import json
from torchvision.datasets import Food101

def main():
	dataset = Food101(root='data/',split='train',download=True)
	
	class_name = dataset.classes
	output_path = os.path.join('data', 'class_names.json')
 
	with open(output_path, 'w', encoding='utf-8') as f:
		json.dump(class_name, f, ensure_ascii=False, indent=4)
	print(f"Class names extracted and saved to {output_path}")

if __name__ == "__main__":
    main()