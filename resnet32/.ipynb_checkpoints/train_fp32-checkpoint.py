import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.cifar_resnet32 import ResNet32


def get_dataloaders(batch_size=128, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616))
    ])

    train_set = datasets.CIFAR10(root="./data", train=True,
                                 download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    correct, total, running_loss = 0, 0, 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total, running_loss = 0, 0, 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100. * correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    train_loader, test_loader = get_dataloaders()

    model = ResNet32(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.1, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 120, 160], gamma=0.2
    )

    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    num_epochs = 100  # reduce to 20 while testing

    # ---- metric logging for plots ----
    metrics = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        te_loss, te_acc = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()

        # log metrics for this epoch
        metrics["train_loss"].append(tr_loss)
        metrics["test_loss"].append(te_loss)
        metrics["train_acc"].append(tr_acc)
        metrics["test_acc"].append(te_acc)

        end = time.time()
        print(f"[{epoch:03}/{num_epochs}] Train Acc: {tr_acc:.2f}% | "
              f"Test Acc: {te_acc:.2f}% | Time: {end-start:.1f}s",
              flush=True)

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(),
                       "checkpoints/resnet32_fp32_best.pt")
            print(f"  → Saved new best: {best_acc:.2f}%", flush=True)

    # save metrics dict for plotting later
    torch.save(metrics, "checkpoints/resnet32_fp32_metrics.pt")
    print("Saved metrics to checkpoints/resnet32_fp32_metrics.pt")
    print("Training done. Best test acc:", best_acc)


if __name__ == "__main__":
    import sys
    sys.stdout.flush()
    main()