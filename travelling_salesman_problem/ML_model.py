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

def create_dataloader(data_dir, batch_size=64, split='train', shuffle=True, num_workers=0):
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

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.attention_block = AttentionBlock(embed_dim, num_heads)
        self.ff_block = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.attention_block(x)
        ff_output = self.ff_block(x)
        x = x + self.dropout(ff_output)
        x = self.norm(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.shortcut = nn.Sequential()
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )

    def forward(self, x):
        out = self.block(x)
        out += self.shortcut(x)
        return nn.ReLU()(out)


class TSMPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=4, num_heads=8, ff_dim=512):
        """
        Args:
            input_dim (int): Dimension of the input feature vector.
            hidden_dim (int): Dimension of the hidden layers.
            output_dim (int): Dimension of the output feature vector.
            num_blocks (int): Number of residual blocks to include.
            num_heads (int): Number of attention heads in the transformer block.
            ff_dim (int): Dimension of the feedforward network in the transformer block.
        """
        super(TSMPModel, self).__init__()

        self.flatten = nn.Flatten()
        self.initial_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)]
        )

        self.transformer_block = TransformerBlock(hidden_dim, num_heads, ff_dim)
        
        self.fc_final = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.initial_block(x)
        x = self.blocks(x)
        x = x.unsqueeze(0)  # Add sequence dimension for transformer (batch_size, seq_len, embed_dim)
        x = self.transformer_block(x)
        x = x.squeeze(0)  # Remove sequence dimension
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
    hidden_dim = 128*4  # You can adjust this parameter based on your model complexity

    model = TSMPModel(input_dim, hidden_dim, output_dim, num_blocks=16,num_heads=32).to(device)

    # Load the model
    model.load_state_dict(torch.load('travelling_salesman_problem/models/model.pth'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    print(f'Model: {model}')
    print(f'Input: {input_obj}')
    print(f'Output: {output_obj}')
    print(f'Input dimension: {input_dim}')
    print(f'Output dimension: {output_dim}')

    train(model, train_dataloader, val_dataloader, test_dataloader, optimizer, loss_fn, epochs=100, device=device)

    #Save the model
    torch.save(model.state_dict(), 'travelling_salesman_problem/models/model.pth')

if __name__ == '__main__':
    main()
