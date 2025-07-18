import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



def load_fashion_mnist_data(batch_size=64):
    """
    Function that loads fashionMNIST dataset
    
    Args:
        batch_size(int) : The number of samples per batch
    Returns:
        tuple : A tuple containing (train_loader, test_loader, device)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device : {device}")
    if device.type == 'cuda':
        print(f"GPU Name : {torch.cuda.get_device_name(0)}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    num_workers = os.cpu_count - 1 if os.cpu_count > 1 else 0
    if num_workers > 4:
        num_workers == 4
    print(f"Using {num_workers} cpu cores")

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True if device.type == 'cuda' else False
    )

    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True if device.type == 'cuda' else False
    )

    return train_loader, test_loader, device