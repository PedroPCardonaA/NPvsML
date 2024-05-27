import torch
from torch import nn
import numpy as np

class ML_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ML_model, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x
    
# Load the trained model
def load_city_map(filename):
    return np.load(filename)

def define_model(filename):
    city_map = load_city_map(filename)
    input_size = city_map.shape[0]
    hidden_size = city_map.shape[0] * 8
    output_size = city_map.shape[0]
    return ML_model(input_size, hidden_size, output_size)


def main():
    model = define_model('travelling_salesman_problem/maps/map_22_f991b792-8782-4073-b869-9d92d948e5db.npy')
    print(model)
    city_map = load_city_map('travelling_salesman_problem/maps/map_22_f991b792-8782-4073-b869-9d92d948e5db.npy')
    print(city_map)
    best_path = model(torch.tensor(city_map).float())
    print(best_path)