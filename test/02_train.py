# scripts/02_train.py
import sys
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Bổ sung thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conf.config import load_config
from utils.reproducibility import set_global_seed
from dataset.builder import Food101DataBuilder
from models.model_factory import ModelFactory

def train_step(model, dataloader, criterion, optimizer, scaler, device):
    """Thực thi một epoch huấn luyện với Automatic Mixed Precision (AMP)."""
    model.train()
    running_loss, running_corrects = 0.0, 0

    # Sử dụng tqdm để hiển thị thanh tiến trình trực quan
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass với AMP để tối ưu VRAM và tốc độ
        with autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward pass qua GradScaler để tránh hiện tượng underflow của float16
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Tính toán độ đo
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)
    return epoch_loss, epoch_acc

def val_step(model, dataloader, criterion, device):
    """Thực thi một epoch đánh giá mô hình."""
    model.eval()
    running_loss, running_corrects = 0.0, 0

    pbar = tqdm(dataloader, desc="Validation", leave=False)
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)
    return epoch_loss, epoch_acc

def main():
    print("[INFO] Khởi tạo quá trình huấn luyện...")
    config = load_config("conf/config.yaml")
    set_global_seed(config.training.seed)

    # 1. Thiết lập thiết bị tính toán
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Sử dụng thiết bị: {device}")

    # 2. Khởi tạo dữ liệu
    data_builder = Food101DataBuilder(config.dataset)
    train_loader, val_loader = data_builder.get_dataloaders()
    print(f"[INFO] Dữ liệu tải thành công. Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 3. Khởi tạo mô hình
    model = ModelFactory.create_model(config.active_model_config)
    model = model.to(device)

    # 4. Cấu hình hàm mất mát, bộ tối ưu hóa và AMP Scaler
    criterion = nn.CrossEntropyLoss()
    # Sử dụng AdamW thay cho Adam truyền thống để tối ưu hóa weight decay tốt hơn
    optimizer = AdamW(
        model.parameters(), 
        lr=config.training.learning_rate, 
        weight_decay=config.training.weight_decay
    )
    scaler = GradScaler() 

    # 5. Khởi tạo thư mục lưu trữ Model Checkpoint
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')

    # 6. Vòng lặp huấn luyện chính
    print(f"[INFO] Bắt đầu huấn luyện: {config.active_model_config.model_name}")
    start_time = time.time()

    for epoch in range(config.training.epochs):
        print(f"\nEpoch {epoch+1}/{config.training.epochs}")
        print("-" * 20)

        train_loss, train_acc = train_step(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = val_step(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Cơ chế lưu mô hình tốt nhất (Best Checkpointing)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(
                config.training.checkpoint_dir, 
                f"{config.active_model_config.model_name}_best.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"[*] Đã lưu checkpoint mới tại: {checkpoint_path}")

    total_time = time.time() - start_time
    print(f"\n[SUCCESS] Hoàn thành huấn luyện trong {total_time/60:.2f} phút.")

if __name__ == "__main__":
    main()