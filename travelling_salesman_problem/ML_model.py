import torch
from torch import nn
import numpy as np

class PointerNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PointerNet, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.pointer = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()

        encoder_outputs, (hidden, cell) = self.encoder(inputs)

        decoder_input = torch.zeros(batch_size, 1, inputs.size(2)).to(inputs.device)
        decoder_hidden = hidden
        decoder_cell = cell
 
        pointers = []
        
        for _ in range(seq_len):
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, (decoder_hidden, decoder_cell))
            pointer_logits = self.pointer(decoder_output).squeeze(-1)
            pointer = torch.argmax(pointer_logits, dim=1)
            pointers.append(pointer)

            decoder_input = torch.gather(inputs, 1, pointer.view(-1, 1, 1).expand(-1, 1, inputs.size(2)))
        
        pointers = torch.stack(pointers).permute(1, 0)
        return pointers

def load_city_map(filename):
    return np.load(filename)

def preprocess_data(city_map):
    return city_map

def define_model(city_map):
    input_size = city_map.shape[-1]
    hidden_size = 128
    model = PointerNet(input_size, hidden_size)
    return model

def main():
    filename = 'travelling_salesman_problem/maps/map_22_f991b792-8782-4073-b869-9d92d948e5db.npy'
    city_map = load_city_map(filename)
    city_map = preprocess_data(city_map)
    city_map = torch.tensor(city_map).float().unsqueeze(0)

    model = define_model(city_map)
    model.eval()  
    with torch.no_grad():  
        best_path = model(city_map)
    print("Best path (indices):", best_path)
    best_path = best_path.squeeze().tolist()
    print("Best path:", best_path)

if __name__ == '__main__':
    main()
