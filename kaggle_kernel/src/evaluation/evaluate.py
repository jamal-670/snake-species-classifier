import os, torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

@torch.no_grad()
def full_evaluation(model, test_loader, class_names, device):
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(test_loader, desc="Evaluating"):
        logits = model(imgs.to(device))
        preds  = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    print("\n=== Classification Report ===")
    print(classification_report(
        all_labels, all_preds,
        target_names=class_names, digits=4))

    # Confusion matrix
    cm  = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues", ax=ax, annot_kws={"size":7})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/confusion_matrix.png", dpi=150)
    plt.close()
    print("Confusion matrix saved → reports/confusion_matrix.png")
