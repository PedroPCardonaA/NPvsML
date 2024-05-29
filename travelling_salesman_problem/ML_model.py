import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from torchdata.datapipes.iter import IterableWrapper

class TSPDataParser(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        Args:
            data_dir (string): Directory with all the numpy files.
            split (string): One of 'train', 'val', 'test' to specify the data split.
        """
        self.data_dir = data_dir
        self.split = split
        self.X_dir = os.path.join(data_dir, 'X-data', split)
        self.Y_dir = os.path.join(data_dir, 'Y-data', split)
        
        # Get list of all file names (assuming both X and Y have same file names)
        self.file_names = [f for f in os.listdir(self.X_dir) if os.path.isfile(os.path.join(self.X_dir, f))]
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get file name
        file_name = self.file_names[idx]
        
        # Load X data
        X_path = os.path.join(self.X_dir, file_name)
        X = np.load(X_path)
        X = torch.tensor(X, dtype=torch.float32)
        
        # Load Y data
        Y_path = os.path.join(self.Y_dir, file_name)
        Y = np.load(Y_path)
        Y = torch.tensor(Y, dtype=torch.float32)
        
        return X, Y

def create_dataloader(data_dir, batch_size=32, split='train', shuffle=True, num_workers=0):
    """
    Args:
        data_dir (string): Directory with all the numpy files.
        batch_size (int): Batch size for the dataloader.
        split (string): One of 'train', 'val', 'test' to specify the data split.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of workers for data loading.
        
    Returns:
        DataLoader: A PyTorch DataLoader for the specified data split.
    """
    dataset = TSPDataParser(data_dir, split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


