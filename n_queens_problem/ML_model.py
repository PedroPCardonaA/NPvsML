import torch
import torch.nn as nn
import torch.nn.functional as F
from Path_generator import get_path
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from Training import train


class NQueensDataParser(Dataset):
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
        
        self.file_names = [f for f in os.listdir(self.X_dir) if os.path.isfile(os.path.join(self.X_dir, f))]
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_name = self.file_names[idx]
        X_path = os.path.join(self.X_dir, file_name)
        X = np.load(X_path)
        X = torch.tensor(X, dtype=torch.float32)
        Y_path = os.path.join(self.Y_dir, file_name)
        Y = np.load(Y_path)
        Y = torch.tensor(Y, dtype=torch.float32)
        return X, Y
    
def create_dataloader(data_dir, batch_size=64, split='train', shuffle=True, num_workers=0):
    dataset = NQueensDataParser(data_dir, split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

class NQueensDataParser(Dataset):
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
        
        self.file_names = [f for f in os.listdir(self.X_dir) if os.path.isfile(os.path.join(self.X_dir, f))]
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_name = self.file_names[idx]
        X_path = os.path.join(self.X_dir, file_name)
        X = np.load(X_path)
        X = torch.tensor(X, dtype=torch.float32)
        Y_path = os.path.join(self.Y_dir, file_name)
        Y = np.load(Y_path)
        Y = torch.tensor(Y, dtype=torch.float32)
        return X, Y
    
def create_dataloader(data_dir, batch_size=64, split='train', shuffle=True, num_workers=0):
    dataset = NQueensDataParser(data_dir, split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class FullyConnectedBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob=0.5):
        super(FullyConnectedBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class NQueensModel(nn.Module):
    def __init__(self, N):
        super(NQueensModel, self).__init__()
        self.N = N
        
        # Define convolutional blocks
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        
        # Calculate the flatten size after convolutional layers
        self.flatten_size = 256 * N * N
        
        # Define fully connected blocks
        self.fc_block1 = FullyConnectedBlock(self.flatten_size, 1024)
        self.fc_block2 = FullyConnectedBlock(1024, 512)
        
        # Final fully connected layer without dropout
        self.fc_final = nn.Linear(512, N * N)
        
    def forward(self, x):
        # Reshape input to match the expected dimensions (batch_size, channels, height, width)
        x = x.view(-1, 1, self.N, self.N)
        
        # Apply convolutional blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # Flatten the output from the convolutional layers
        x = x.view(-1, self.flatten_size)
        
        # Apply fully connected blocks
        x = self.fc_block1(x)
        x = self.fc_block2(x)
        
        # Apply final fully connected layer to get the output
        x = self.fc_final(x)
        
        # Reshape the output to match the desired output dimensions (batch_size, N, N)
        x = x.view(-1, self.N, self.N)
        
        # Apply sigmoid activation to get values between 0 and 1
        x = torch.sigmoid(x)
        
        return x

def define_model():
    path = get_path()
    data_dir = f'{path}/boards'
    train_dataloader = create_dataloader(data_dir, split='train')
    one_batch = next(iter(train_dataloader))
    input_obj, output_obj = one_batch
    input_dim = input_obj.view(input_obj.shape[0], -1).shape[1]  # Flatten the input to 1D
    output_dim = output_obj.view(output_obj.shape[0], -1).shape[1]  # Flatten the output to 1D
    hidden_dim = 128*4  # You can adjust this parameter based on your model complexity
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NQueensModel(input_dim)
    return model    


def main():
    path = get_path()
    data_dir = f'{path}/boards'
    train_dataloader = create_dataloader(data_dir, split='train')
    val_dataloader = create_dataloader(data_dir, split='val')
    test_dataloader = create_dataloader(data_dir, split='test')
    model = define_model()

    # model.load_state_dict(torch.load(f'{path}/models/model.pth'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    train(model, train_dataloader, val_dataloader, test_dataloader, optimizer, loss_fn, epochs=100, device=device)

    #Save the model
    torch.save(model.state_dict(), f'{path}/models/model.pth')

if __name__ == '__main__':
    main()



