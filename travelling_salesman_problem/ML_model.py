import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from torchdata.datapipes.iter import IterableWrapper
from Training import train


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
        super(TSMPModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(output_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, src_len, _ = src.size()
        trg_len = trg.size(1)
        trg_vocab_size = self.out.out_features

        encoder_outputs, (hidden, cell) = self.encoder(src)
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)
        input = trg[:, 0, :]

        for t in range(1, trg_len):
            output, (hidden, cell) = self.decoder(input.unsqueeze(1), (hidden, cell))
            attn_weights = torch.bmm(encoder_outputs, hidden.unsqueeze(2)).squeeze(2)
            attn_weights = torch.softmax(attn_weights, dim=1)
            context = torch.bmm(encoder_outputs.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)
            output = torch.tanh(self.attention(torch.cat((context, hidden.squeeze(0)), dim=1)))
            output = self.out(output)
            outputs[:, t, :] = output
            teacher_force = np.random.random() < teacher_forcing_ratio
            input = trg[:, t, :] if teacher_force else output

        return outputs
    
    
def main():
    data_dir = 'travelling_salesman_problem/maps'
    train_dataloader = create_dataloader(data_dir, split='train')
    val_dataloader = create_dataloader(data_dir, split='val')
    test_dataloader = create_dataloader(data_dir, split='test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_dim = train_dataloader.dataset[0][0].shape[1]
    hidden_dim = 128
    output_dim = train_dataloader.dataset[0][1].shape[1] + 2

    model = TSMPModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train(model, train_dataloader, val_dataloader, test_dataloader, optimizer, epochs=5, device=device)

    # Save the model
    torch.save(model.state_dict(), 'travelling_salesman_problem/maps/model.pth')


if __name__ == '__main__':
    main()