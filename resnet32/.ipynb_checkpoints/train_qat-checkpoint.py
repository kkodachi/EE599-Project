import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.cifar_resnet32 import ResNet32

from torch.ao.quantization import (
    get_default_qat_qconfig,
    prepare_qat,
    convert,
)


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
    # ------- setup -------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("QAT running on:", device)

    train_loader, test_loader = get_dataloaders()

    # ------- start from FP32 baseline weights -------
    fp32_ckpt = "checkpoints/resnet32_fp32_best.pt"
    assert os.path.exists(fp32_ckpt), f"Missing {fp32_ckpt}, run train_fp32.py first."

    float_model = ResNet32(num_classes=10)
    float_model.load_state_dict(torch.load(fp32_ckpt, map_location="cpu"))
    print("Loaded FP32 baseline from", fp32_ckpt)

    # ------- configure QAT -------
    # backend for x86 quantization; training can still run on CUDA
    torch.backends.quantized.engine = "fbgemm"

    # attach a default QAT qconfig
    float_model.qconfig = get_default_qat_qconfig("fbgemm")
    print("QAT qconfig:", float_model.qconfig)

    # prepare_qat inserts fake-quant + observers
    qat_model = prepare_qat(float_model)
    qat_model.to(device)

    # ------- QAT training setup -------
    criterion = nn.CrossEntropyLoss()
    # smaller LR than FP32, we are fine-tuning
    optimizer = optim.SGD(qat_model.parameters(),
                          lr=1e-3, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[15, 25], gamma=0.1
    )

    num_epochs = 20  # QAT fine-tuning epochs

    os.makedirs("checkpoints", exist_ok=True)

    metrics = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    best_fakeq_acc = 0.0

    # ------- QAT training loop (fake-quant) -------
    for epoch in range(1, num_epochs + 1):
        start = time.time()

        tr_loss, tr_acc = train_one_epoch(
            qat_model, train_loader, criterion, optimizer, device
        )
        te_loss, te_acc = evaluate(
            qat_model, test_loader, criterion, device
        )
        scheduler.step()

        metrics["train_loss"].append(tr_loss)
        metrics["test_loss"].append(te_loss)
        metrics["train_acc"].append(tr_acc)
        metrics["test_acc"].append(te_acc)

        end = time.time()
        print(f"[QAT {epoch:03}/{num_epochs}] Train Acc: {tr_acc:.2f}% | "
              f"Test Acc (fake-quant): {te_acc:.2f}% | Time: {end-start:.1f}s",
              flush=True)

        if te_acc > best_fakeq_acc:
            best_fakeq_acc = te_acc
            torch.save(qat_model.state_dict(),
                       "checkpoints/resnet32_qat_fake_best.pt")
            print(f"  → Saved new best fake-quant model: {best_fakeq_acc:.2f}%", flush=True)

    # save metrics for plotting
    torch.save(metrics, "checkpoints/resnet32_qat_metrics.pt")
    print("Saved QAT metrics to checkpoints/resnet32_qat_metrics.pt")

    # ------- convert to real INT8 model on CPU -------
    qat_model.cpu()
    int8_model = convert(qat_model.eval())
    torch.save(int8_model.state_dict(), "checkpoints/resnet32_int8_qat.pt")
    print("Saved converted INT8-QAT model to checkpoints/resnet32_int8_qat.pt")



if __name__ == "__main__":
    import sys
    sys.stdout.flush()
    main()