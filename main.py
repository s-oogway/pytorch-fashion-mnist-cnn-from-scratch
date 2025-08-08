import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.cnn_model import CNNModel



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

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
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

    num_workers = os.cpu_count() - 1 if os.cpu_count() > 1 else 0
    if num_workers > 4:
        num_workers = 4
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


def train_model(model, train_loader, criterion, optimizer, device):
    """
    Trains model (one epoch for now)
    Args: 
        model : The model that is being trained
        train_loader: The dataloader for training data
        criterion: The loss function that will be used
        optimizer: The optimizer that will be used
        device: The device the model will be trained on
    """
    pass

def main():
    """
    Main function to (do initial data loading test)
    """
    print("Starting data loading test...")
    train_loader, test_loader, device = load_fashion_mnist_data()
    images, labels = next(iter(train_loader))

    # Instantiate model and move to device
    model = CNNModel().to(device)
    print("\n--- Model Architecture ---")
    print(model)

    # Pass the batch through the model
    # Ensure the images tensor is on the same device as the model
    outputs = model(images.to(device))
    print(f"\nModel output shape: {outputs.shape}")

if __name__ == '__main__':
    main()