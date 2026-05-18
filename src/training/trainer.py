import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

def smooth_loss(logits, targets, num_classes, smoothing=0.1):
    confidence = 1.0 - smoothing
    log_probs  = nn.functional.log_softmax(logits, dim=-1)
    nll        = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    smooth     = -log_probs.mean(dim=-1)
    return (confidence * nll + smoothing * smooth).mean()

def run_epoch(model, loader, optimizer, scaler, device,
              num_classes, training=True):
    model.train() if training else model.eval()
    total_loss = correct = total = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
                with autocast():
                    logits = model(imgs)
                    loss   = smooth_loss(logits, labels, num_classes)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss   = smooth_loss(logits, labels, num_classes)

            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1)==labels).sum().item()
            total      += imgs.size(0)

    return total_loss/total, correct/total

def run_training(model, train_loader, val_loader,
                 config, device, resume_from=None):
    os.makedirs("checkpoints", exist_ok=True)
    num_classes = config["num_classes"]
    tr          = config["training"]
    scaler      = GradScaler()
    best_acc    = 0
    start_epoch = 0

    if resume_from and os.path.exists(resume_from):
        ckpt        = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        best_acc    = ckpt.get("val_acc", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}")

    stages = [
        ("Stage 1 — head only",
         tr["stage1_epochs"], tr["stage1_lr"], "freeze"),
        ("Stage 2 — top blocks",
         tr["stage2_epochs"], tr["stage2_lr"], "top"),
        ("Stage 3 — full model",
         tr["stage3_epochs"], tr["stage3_lr"], "all"),
    ]

    model.to(device)
    global_epoch = 0

    for stage_name, epochs, lr, unfreeze in stages:
        print(f"\n{'='*50}")
        print(f"  {stage_name}  |  lr={lr}  |  epochs={epochs}")
        print(f"{'='*50}")

        if unfreeze == "freeze": model.freeze_backbone()
        elif unfreeze == "top":  model.unfreeze_top(3)
        else:                    model.unfreeze_all()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=tr["weight_decay"])
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        patience  = 0

        for epoch in range(epochs):
            if global_epoch < start_epoch:
                global_epoch += 1
                continue

            tr_loss, tr_acc = run_epoch(
                model, train_loader, optimizer, scaler,
                device, num_classes, training=True)
            vl_loss, vl_acc = run_epoch(
                model, val_loader, optimizer, scaler,
                device, num_classes, training=False)
            scheduler.step()

            print(f"  Epoch {global_epoch+1:03d} | "
                  f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
                  f"val_loss={vl_loss:.4f} acc={vl_acc:.3f}")

            if vl_acc > best_acc:
                best_acc = vl_acc
                torch.save({
                    "epoch":       global_epoch,
                    "model_state": model.state_dict(),
                    "val_acc":     best_acc,
                    "num_classes": num_classes,
                }, "checkpoints/best_model.pt")
                print(f"  ✓ Best model saved (val_acc={best_acc:.4f})")
                patience = 0
            else:
                patience += 1
                if patience >= tr["early_stop_patience"]:
                    print(f"  Early stopping at epoch {global_epoch+1}")
                    break

            global_epoch += 1

    print(f"\nTraining complete. Best val_acc: {best_acc:.4f}")
    return best_acc
