import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

from src.data.datasets import get_cifar100_dataloaders, get_gtsrb_dataloaders, get_vggface_dataloaders, get_chexpert_dataloaders
from src.models.resnet import ResNet18
from src.detector.watermark_monitor import WatermarkMonitor
from src.utils.runtime import resolve_device, get_torch_load_kwargs
from run_phase3 import get_watermarks
from train import train_epoch

def main():
    parser = argparse.ArgumentParser(description="Phase 4 Pipeline: Unauthorized Training Simulation")
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'gtsrb', 'vggface', 'chexpert'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10, help="Epochs to train stolen model (demo)")
    parser.add_argument('--num-poisons', type=int, default=100)
    parser.add_argument('--target-class', type=int, default=0)
    parser.add_argument('--poison-class', type=int, default=1)
    parser.add_argument('--poison-steps', type=int, default=50)
    parser.add_argument('--poison-lr', type=float, default=0.1)
    parser.add_argument('--poison-epsilon', type=float, default=16/255)
    parser.add_argument('--stolen-lr', type=float, default=0.01)
    parser.add_argument('--stolen-weight-decay', type=float, default=5e-4)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--out-dir', type=str, default='./results')
    parser.add_argument('--auth-model-path', type=str, required=True, help="Path to clean baseline model")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = resolve_device(args.device)
    pin_memory = device.type == 'cuda'
    print(f"Using device: {device}")

    # 1. Setup Data and Authorized Model
    print("\n--- Setup: Loading Authorized Environment ---")
    is_32x32 = True
    if args.dataset == 'cifar100':
        train_loader, val_loader, test_loader, full_train_dataset = get_cifar100_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory)
        num_classes = 100
    elif args.dataset == 'gtsrb':
        train_loader, val_loader, test_loader, full_train_dataset = get_gtsrb_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory)
        num_classes = 43
    elif args.dataset == 'vggface':
        train_loader, val_loader, test_loader, full_train_dataset = get_vggface_dataloaders(data_dir=os.path.join(args.data_dir, 'VGGFace2'), batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory)
        num_classes = 8631
        is_32x32 = False
    elif args.dataset == 'chexpert':
        train_loader, val_loader, test_loader, full_train_dataset = get_chexpert_dataloaders(data_dir=os.path.join(args.data_dir, 'CheXpert'), batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory)
        num_classes = 5
        is_32x32 = False
    
    auth_model = ResNet18(num_classes=num_classes, is_32x32=is_32x32).to(device)
    auth_model.load_state_dict(torch.load(args.auth_model_path, **get_torch_load_kwargs(device)))
    auth_model.eval()

    # 2. Generate Watermarks and Get Reference Signatures
    print("\n--- Setup: Extracting Watermark Probes ---")
    watermarks, labels = get_watermarks(
        auth_model,
        full_train_dataset,
        args.num_poisons,
        target_class=args.target_class,
        poison_class=args.poison_class,
        device=device,
        poison_steps=args.poison_steps,
        poison_lr=args.poison_lr,
        poison_epsilon=args.poison_epsilon,
    )
    watermark_loader = DataLoader(TensorDataset(watermarks, labels), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    monitor = WatermarkMonitor(model=auth_model)
    monitor.generate_reference_signatures(watermark_loader, device)

    # 3. Create the "Stolen" Dataset (Clean + Watermarks)
    # In a real scenario, the watermarks would be embedded in the released dataset.
    # We simulate this by training the target model on the clean dataset mixed with watermarks.
    # For simplicity in this script, we'll just train on the clean dataloader and periodically 
    # train on a batch of watermarks to simulate the embedding.
    print("\n--- Phase 4: Simulating Unauthorized Training ---")
    stolen_model = ResNet18(num_classes=num_classes, is_32x32=is_32x32).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(stolen_model.parameters(), lr=args.stolen_lr, momentum=0.9, weight_decay=args.stolen_weight_decay)

    # Tracking metrics
    history = {layer: [] for layer in monitor.layer_names}
    history['epochs'] = list(range(args.epochs + 1))

    # Epoch 0 (Untrained state)
    print("Auditing Untrained Model...")
    alignment = monitor.audit_model(stolen_model, watermark_loader, device)
    for layer, score in alignment.items():
        history[layer].append(score)

    print("\nTraining Stolen Model...")
    for epoch in range(1, args.epochs + 1):
        # 3a. Train on standard data
        train_loss, train_acc = train_epoch(stolen_model, train_loader, criterion, optimizer, device, epoch)
        
        # 3b. Train on the embedded watermarks (simulating they are in the dataset)
        # We explicitly step on the watermarks to ensure the model learns their features
        stolen_model.train()
        for w_imgs, w_labels in watermark_loader:
            optimizer.zero_grad()
            outputs = stolen_model(w_imgs.to(device))
            loss = criterion(outputs, w_labels.to(device))
            loss.backward()
            optimizer.step()

        # 4. Audit the model at the end of the epoch
        alignment = monitor.audit_model(stolen_model, watermark_loader, device)
        for layer, score in alignment.items():
            history[layer].append(score)
            
        print(f"Epoch {epoch} Alignment -> {alignment}")

    # 5. Visualize the tracking metrics (Phase 5)
    print("\n--- Phase 5: Generating Tracing Report ---")
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    for layer in monitor.layer_names:
        plt.plot(history['epochs'], history[layer], marker='o', label=layer)
        
    dataset_title = args.dataset.upper() if args.dataset != 'cifar100' else 'CIFAR-100'
    plt.title(f"Latent Space Alignment of Stolen Model During Training ({dataset_title})")
    plt.xlabel("Training Epochs")
    plt.ylabel("Cosine Similarity to Authorized Signature")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    
    report_path = os.path.join(args.out_dir, f"watermark_tracing_report_{args.dataset}.png")
    plt.savefig(report_path)
    print(f"Saved tracing visualization to {report_path}")

if __name__ == '__main__':
    main()
