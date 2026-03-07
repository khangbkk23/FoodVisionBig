import sys
import os
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conf.config import load_config
from utils.reproducibility import set_global_seed
from dataset.builder import Food101DataBuilder
from models.model_factory import ModelFactory


def train_step(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, running_corrects = 0.0, 0

    pbar = tqdm(dataloader, desc="Train", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels).item()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def val_step(model, dataloader, criterion, device):
    model.eval()
    running_loss, running_corrects = 0.0, 0

    pbar = tqdm(dataloader, desc="Val", leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def main():
    print("[INFO] Initializing training...")

    config = load_config("conf/config.yaml")
    set_global_seed(config.training.seed)

    device = torch.device(
        config.training.device if torch.cuda.is_available() else "cpu"
    )

    print(f"[INFO] Device: {device}")

    data_builder = Food101DataBuilder(config.dataset)
    train_loader, val_loader = data_builder.get_dataloaders()

    model = ModelFactory.create_model(config.active_model_config)
    
    # [TÙY CHỌN BỔ SUNG: KHÔI PHỤC CHECKPOINT NẾU BẬT FINE-TUNE]
    if getattr(config.active_model_config, 'fine_tune', False):
        checkpoint_path = os.path.join(
            config.training.checkpoint_dir,
            f"{config.active_model_config.model_name}_best.pth"
        )
        if os.path.exists(checkpoint_path):
            print(f"[INFO] Loading best parameter for Fine-Tuning: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False) 
        else:
            print("[WARN] Cannot found checkpoint for Fine-Tuning. Training from scratch...")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    ) 

    scaler = GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )

    os.makedirs(config.training.checkpoint_dir, exist_ok=True)

    eval_dir = "evaluates"
    os.makedirs(eval_dir, exist_ok=True)

    history_file_path = os.path.join(
        eval_dir,
        f"history_{config.active_model_config.model_name}.json"
    )

    best_val_loss = float("inf")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rate": [], 
        "initial_lr": config.training.learning_rate, 
        "model_name": config.active_model_config.model_name,
        "epochs": config.training.epochs,
    }

    print(f"[INFO] Model: {config.active_model_config.model_name}")
    print(f"[INFO] Epochs: {config.training.epochs}")
    print("-" * 40)

    start_time = time.time()

    for epoch in range(config.training.epochs):

        print(f"\nEpoch [{epoch+1}/{config.training.epochs}]")

        train_loss, train_acc = train_step(
            model, train_loader, criterion, optimizer, scaler, device
        )

        val_loss, val_acc = val_step(
            model, val_loader, criterion, device
        )

        print(
            f"Train | loss: {train_loss:.4f} | acc: {train_acc:.4f}"
        )
        print(
            f"Val   | loss: {val_loss:.4f} | acc: {val_acc:.4f}"
        )
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history["learning_rate"].append(current_lr)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        with open(history_file_path, "w") as f:
            json.dump(history, f, indent=4)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint_path = os.path.join(
                config.training.checkpoint_dir,
                f"{config.active_model_config.model_name}_best.pth"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )

            print(f"[SAVE] New best model -> {checkpoint_path}")

    total_time = time.time() - start_time

    print("\n[FINISHED] Training completed")
    print(f"[TIME] {total_time/60:.2f} minutes")
    print(f"[HISTORY] {history_file_path}")

if __name__ == "__main__":
    main()