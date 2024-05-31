import torch
from ML_model import TSMPModel, define_model
from NP_algorithm import load_map, plot_path, held_karp
import time
from Path_generator import get_path

def test_model(model_path, map_path, model: TSMPModel = None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    # Load model
    model.load_state_dict(torch.load(model_path))
    
    # Load map
    cities = load_map(map_path)
    
    # Predict path
    time_start = time.time()
    X = torch.tensor(cities, dtype=torch.float32).unsqueeze(0)
    X = X.to(device)
    
    path = model(X)
    path = path.squeeze().detach().cpu().numpy()
    time_end = time.time()
    print(f'Best path: {path}')
    print(f'Time taken: {time_end - time_start:.2f} seconds')
    
    # Plot path
    plot_path(cities, path)

def test(path):
    model_path = f'{path}/models/model.pth'
    map_path = f'{path}/maps/map_18_24546156-60e7-4150-85f9-03c86975700e.npy'
    cities = load_map(map_path)

    print('Held-Karp Algorithm:')
    
    start_time = time.time()
    result = held_karp(cities)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    path, distance = result[:-1], result[-1]
    
    print(f'Best path: {path}')
    print(f'Distance: {distance}')
    print(f'Time taken: {elapsed_time:.2f} seconds')
    
    plot_path(cities, result)

    print('Neural Network Model:')
    model = define_model()
    test_model(model_path, map_path, model)

def main():
    path = get_path()
    test(path)

if __name__ == '__main__':
    main()
