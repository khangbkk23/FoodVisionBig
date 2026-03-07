import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.builder import Food101DataBuilder
from conf.config import DatasetConfig

def test_dataloader_integrity():
    print("[INFO] Loading mock config via dataclass...")
    mock_config = DatasetConfig(
        data_dir='data/',
        batch_size=32,
        num_workers=2,
        pin_memory=False=
    )

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

    print("[INFO] Running assertions...")
    assert images.shape == (mock_config.batch_size, 3, mock_config.image_size, mock_config.image_size), "Invalid shape!"
    assert labels.shape == (mock_config.batch_size,), "Invalid label shape!"
    assert isinstance(images, torch.Tensor), "Not a PyTorch Tensor!"
    assert images.dtype == torch.float32, "Expected float32!"
    assert labels.dtype == torch.long, "Expected long!"

    print("[SUCCESS] DataLoader integration test passed.")

if __name__ == "__main__":
    with torch.no_grad():
        test_dataloader_integrity()