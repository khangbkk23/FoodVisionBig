# scripts/03_evaluate.py
import sys
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conf.config import load_config
from dataset.builder import Food101DataBuilder
from models.model_factory import ModelFactory

def get_predictions(model, dataloader, device):
    """Executes full-dataset inference and returns true/predicted labels."""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return np.array(all_labels), np.array(all_preds)

def plot_learning_curves(history_path, save_dir, model_name):
    if not os.path.exists(history_path):
        print(f"History file missing: {history_path}")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Subplot
    ax1.plot(epochs, history["train_loss"], label='Train Loss', marker='o')
    ax1.plot(epochs, history["val_loss"], label='Val Loss', marker='o')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Accuracy Subplot
    ax2.plot(epochs, history["train_acc"], label='Train Acc', marker='o')
    ax2.plot(epochs, history["val_acc"], label='Val Acc', marker='o')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    curve_path = os.path.join(save_dir, f"learning_curve_{model_name}.png")
    plt.savefig(curve_path, dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_dir, model_name):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(40, 40))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d", 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title("Confusion Matrix", fontsize=30)
    plt.ylabel("True Label", fontsize=20)
    plt.xlabel("Predicted Label", fontsize=20)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    cm_path = os.path.join(save_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("[INFO] Initializing evaluation...")
    
    config = load_config("conf/config.yaml")
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    model_name = config.active_model_config.model_name

    data_builder = Food101DataBuilder(config.dataset)
    _, val_loader = data_builder.get_dataloaders()
    class_names = val_loader.dataset.classes

    checkpoint_path = os.path.join(config.training.checkpoint_dir, f"{model_name}_best.pth")
    if not os.path.exists(checkpoint_path):
        print(f"[ERR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    model = ModelFactory.create_model(config.active_model_config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    eval_dir = "evaluates"
    os.makedirs(eval_dir, exist_ok=True)
    # Learning curve
    history_file_path = os.path.join(eval_dir, f"history_{model_name}.json")
    plot_learning_curves(history_file_path, eval_dir, model_name)

    y_true, y_pred = get_predictions(model, val_loader, device)
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    report_path = os.path.join(eval_dir, f"classification_report_{model_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report) 
    # Confusion matrix plotting
    plot_confusion_matrix(y_true, y_pred, class_names, eval_dir, model_name)
    
    print(f"[SUCCESS] Evaluation finished. Artifacts saved in '{eval_dir}/'")

if __name__ == "__main__":
    main()