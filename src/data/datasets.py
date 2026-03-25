import os
import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_cifar100_dataloaders(data_dir='./data', batch_size=128, val_split=0.1, num_workers=4):
    """
    Downloads and prepares CIFAR-100 dataloaders.
    Applies standard data augmentation for training.
    """
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    full_train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_dataset_clean = datasets.CIFAR100(root=data_dir, train=True, download=False, transform=test_transform)
    val_dataset.dataset = val_dataset_clean

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, full_train_dataset

def get_gtsrb_dataloaders(data_dir='./data', batch_size=128, val_split=0.1, num_workers=4):
    """
    Downloads and prepares GTSRB dataloaders.
    Images are resized to 32x32 to match CIFAR architectures.
    """
    mean = (0.3337, 0.3064, 0.3171)
    std = (0.2672, 0.2564, 0.2629)

    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    full_train_dataset = datasets.GTSRB(root=data_dir, split='train', download=True, transform=train_transform)
    test_dataset = datasets.GTSRB(root=data_dir, split='test', download=True, transform=test_transform)

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_dataset_clean = datasets.GTSRB(root=data_dir, split='train', download=False, transform=test_transform)
    val_dataset.dataset = val_dataset_clean

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, full_train_dataset

def get_vggface_dataloaders(data_dir='./data/VGGFace2', batch_size=64, num_workers=4):
    """
    Prepares VGGFace2 dataloaders.
    Requires 224x224 input size for standard architectures.
    """
    # Standard ImageNet normalization for ResNet architectures
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dir = os.path.join(data_dir, 'train')

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"VGGFace2 'train' directory not found in {data_dir}.")

    full_train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    
    # The original 'val' directory in VGGFace2 often contains entirely disjoint classes from 'train'.
    # If we evaluate on disjoint classes, accuracy will be 0%.
    # Therefore, we split the training set into train and validation sets (90/10 split)
    val_size = int(len(full_train_dataset) * 0.1)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply validation transforms to the validation split
    val_dataset_clean = datasets.ImageFolder(root=train_dir, transform=val_transform)
    val_dataset.dataset = val_dataset_clean
    
    # We will use the validation set as the test set for this implementation
    test_dataset = val_dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, full_train_dataset

class CheXpertDataset(Dataset):
    """
    Custom Dataset for CheXpert.
    Parses the CSV and loads images efficiently.
    For simplicity in this baseline, we map the task to a single dominant pathology prediction (e.g. Pleural Effusion)
    to keep the CrossEntropy pipeline identical across datasets, rather than switching to BCEWithLogitsLoss.
    """
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Read the CSV
        self.df = pd.read_csv(csv_file)
        
        # Fill missing values with 0 and uncertain (-1) with 0 for simplicity in this baseline
        self.df = self.df.fillna(0)
        self.df = self.df.replace(-1.0, 0.0)
        
        # We will use a subset of 5 core pathologies as classes for multi-class classification
        # where we pick the 'most severe' or first positive one as the label to fit the existing pipeline.
        self.pathologies = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        
        # Filter dataframe to only include rows that have at least one of these positive
        self.df['has_pathology'] = self.df[self.pathologies].max(axis=1)
        self.df = self.df[self.df['has_pathology'] == 1.0].reset_index(drop=True)
        
        # Create a single label (0-4) based on the first positive pathology found
        # This is a simplification to allow the standard pipeline to run without changing the loss function
        def get_single_label(row):
            for i, p in enumerate(self.pathologies):
                if row[p] == 1.0:
                    return i
            return 0
            
        self.labels = self.df.apply(get_single_label, axis=1).values
        
        # Fix paths. The CSV has 'CheXpert-v1.0-small/train/...' but the root_dir is already 'data/CheXpert'
        # We need to strip the 'CheXpert-v1.0-small/' prefix if it exists so it maps correctly to the extracted folder.
        self.image_paths = self.df['Path'].apply(lambda x: x.replace('CheXpert-v1.0-small/', '')).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        max_attempts = 10
        attempts = 0
        cur_idx = int(idx)

        while attempts < max_attempts:
            img_name = os.path.join(self.root_dir, self.image_paths[cur_idx])
            try:
                image = Image.open(img_name).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                label = int(self.labels[cur_idx])
                return image, label
            except (UnidentifiedImageError, OSError, ValueError):
                attempts += 1
                cur_idx = (cur_idx + 1) % len(self)

        image = Image.new('RGB', (224, 224))
        if self.transform:
            image = self.transform(image)
        return image, int(self.labels[cur_idx])

def get_chexpert_dataloaders(data_dir='./data/CheXpert', batch_size=64, num_workers=4):
    """
    Prepares CheXpert dataloaders.
    Requires 224x224 input size for standard architectures.
    """
    train_csv = os.path.join(data_dir, 'train.csv')
    valid_csv = os.path.join(data_dir, 'valid.csv')
    
    if not os.path.exists(train_csv) or not os.path.exists(valid_csv):
        raise FileNotFoundError(f"CheXpert CSVs not found in {data_dir}. Please ensure 'train.csv' and 'valid.csv' exist.")

    # ImageNet normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    full_train_dataset = CheXpertDataset(csv_file=train_csv, root_dir=data_dir, transform=train_transform)
    test_dataset = CheXpertDataset(csv_file=valid_csv, root_dir=data_dir, transform=val_transform)
    
    # Split train into train/val
    val_size = int(len(full_train_dataset) * 0.1)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create a clean validation dataset
    val_dataset_clean = CheXpertDataset(csv_file=train_csv, root_dir=data_dir, transform=val_transform)
    val_dataset.dataset = val_dataset_clean

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, full_train_dataset

