import torch
from Training import train
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from torch import nn

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

class TSMPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): Dimension of the input feature vector.
            hidden_dim (int): Dimension of the hidden layers.
            output_dim (int): Dimension of the output feature vector.
        """
        super(TSMPModel, self).__init__()

        # Define the model layers
        self.flatten = nn.Flatten()

        self.block1 = self._make_block(input_dim, hidden_dim)
        self.block2 = self._make_block(hidden_dim, hidden_dim)
        self.block3 = self._make_block(hidden_dim, hidden_dim)
        self.block4 = self._make_block(hidden_dim, hidden_dim)

        self.fc_final = nn.Linear(hidden_dim, output_dim)

    def _make_block(self, in_dim, out_dim):
        block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        return block

    def forward(self, x):
        x = self.flatten(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.fc_final(x)

        return x

def main():
    

    data_dir = 'travelling_salesman_problem/maps'
    train_dataloader = create_dataloader(data_dir, split='train')
    val_dataloader = create_dataloader(data_dir, split='val')
    test_dataloader = create_dataloader(data_dir, split='test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    one_batch = next(iter(train_dataloader))
    input_obj, output_obj = one_batch
    input_dim = input_obj.view(input_obj.shape[0], -1).shape[1]  # Flatten the input to 1D
    output_dim = output_obj.view(output_obj.shape[0], -1).shape[1]  # Flatten the output to 1D
    hidden_dim = 128  # You can adjust this parameter based on your model complexity

    model = TSMPModel(input_dim, hidden_dim, output_dim).to(device)

    # Load the model
    # model.load_state_dict(torch.load('travelling_salesman_problem/models/model.pth'))

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    loss_fn = nn.MSELoss()

    print(f'Model: {model}')
    print(f'Input: {input_obj}')
    print(f'Output: {output_obj}')
    print(f'Input dimension: {input_dim}')
    print(f'Output dimension: {output_dim}')

    train(model, train_dataloader, val_dataloader, test_dataloader, optimizer, loss_fn, epochs=1000, device=device)

    #Save the model
    torch.save(model.state_dict(), 'travelling_salesman_problem/models/model.pth')

if __name__ == '__main__':
    main()
