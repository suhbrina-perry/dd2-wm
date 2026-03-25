import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.datasets import get_cifar100_dataloaders, get_gtsrb_dataloaders, get_vggface_dataloaders, get_chexpert_dataloaders
from src.models.resnet import ResNet18
from src.utils.runtime import resolve_device, get_torch_load_kwargs

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
        
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device, epoch, mode="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [{mode}]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
            
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description="Baseline Model Training")
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'gtsrb', 'vggface', 'chexpert'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--out-dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = resolve_device(args.device)
    pin_memory = device.type == 'cuda'
    print(f"Using device: {device}")

    # Load Data
    is_32x32 = True
    if args.dataset == 'cifar100':
        train_loader, val_loader, test_loader, _ = get_cifar100_dataloaders(
            data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory
        )
        num_classes = 100
    elif args.dataset == 'gtsrb':
        train_loader, val_loader, test_loader, _ = get_gtsrb_dataloaders(
            data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory
        )
        num_classes = 43
    elif args.dataset == 'vggface':
        train_loader, val_loader, test_loader, _ = get_vggface_dataloaders(
            data_dir=os.path.join(args.data_dir, 'VGGFace2'), batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory
        )
        num_classes = 8631
        is_32x32 = False
    elif args.dataset == 'chexpert':
        train_loader, val_loader, test_loader, _ = get_chexpert_dataloaders(
            data_dir=os.path.join(args.data_dir, 'CheXpert'), batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory
        )
        num_classes = 5
        is_32x32 = False

    # Initialize Model
    model = ResNet18(num_classes=num_classes, is_32x32=is_32x32).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, f'logs_{args.dataset}'))

    best_acc = float('-inf')
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch, mode="Val")
        
        scheduler.step()
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            print(f"Validation accuracy improved from {best_acc:.2f}% to {val_acc:.2f}%. Saving model...")
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, f'best_{args.dataset}_resnet18.pth'))

    # Final Test
    print("Loading best model for final testing...")
    model.load_state_dict(torch.load(os.path.join(args.out_dir, f'best_{args.dataset}_resnet18.pth'), **get_torch_load_kwargs(device)))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, epoch="Final", mode="Test")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    writer.close()

if __name__ == '__main__':
    main()
