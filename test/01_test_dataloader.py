# test/01_test_dataloader.py
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.builder import Food101DataBuilder

def test_dataloader_integrity():
    print("[INFO] Loading mock config...")
    mock_config = {
        'dataset': {
            'name': 'food101',
            'data_dir': 'data/',
            'image_size': 224,
            'batch_size': 32,
            'num_workers': 2,
            'pin_memory': False,
            'download': True
        }
    }

    print("[INFO] Initializing DataBuilder...")
    try:
        builder = Food101DataBuilder(mock_config)
    except Exception as e:
        print(f"[ERR] DataBuilder init failed: {e}")
        sys.exit(1)

    print("[INFO] Generating DataLoaders...")
    train_loader, val_loader = builder.get_dataloaders()

    print(f"[INFO] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    print("[INFO] Fetching single train batch...")
    try:
        images, labels = next(iter(train_loader))
    except StopIteration:
        print("[ERR] DataLoader is empty!")
        sys.exit(1)

    print(f"[TEST] Image shape: {images.shape}")
    print(f"[TEST] Label shape: {labels.shape}")
    print(f"[TEST] Image dtype: {images.dtype}")
    print(f"[TEST] Label dtype: {labels.dtype}")
    print(f"[TEST] Value range: Min={images.min():.4f}, Max={images.max():.4f}")

    print("[INFO] Running assertions...")
    assert images.shape == (32, 3, 224, 224), "Assertion failed: Invalid image tensor shape!"
    assert labels.shape == (32,), "Assertion failed: Invalid label tensor shape!"
    assert isinstance(images, torch.Tensor), "Assertion failed: Images are not PyTorch Tensors!"
    assert images.dtype == torch.float32, "Assertion failed: Expected float32 for images!"
    assert labels.dtype == torch.long, "Assertion failed: Expected long for labels!"

    print("[SUCCESS] DataLoader integration test passed.")

if __name__ == "__main__":
    with torch.no_grad():
        test_dataloader_integrity()