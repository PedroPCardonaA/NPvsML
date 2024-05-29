import torch
from ML_model import TSMPModel
from NP_algorithm import load_map, plot_path, held_karp
import time

def test_model(model_path, map_path, model: TSMPModel = None):
    model.eval()
    # Load model
    model.load_state_dict(torch.load(model_path))
    
    # Load map
    cities = load_map(map_path)
    
    # Predict path
    time_start = time.time()
    X = torch.tensor(cities, dtype=torch.float32).unsqueeze(0)
    
    path = model(X)
    path = path.squeeze().detach().numpy()
    time_end = time.time()
    print(f'Best path: {path}')
    print(f'Time taken: {time_end - time_start:.2f} seconds')
    
    
    # Plot path
    plot_path(cities, path)

def test():
    model_path = 'travelling_salesman_problem/models/model.pth'
    map_path = 'travelling_salesman_problem/maps/map_18_24546156-60e7-4150-85f9-03c86975700e.npy'
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
    model = TSMPModel(18*18, 128*2, 20)
    model.load_state_dict(torch.load(model_path))
    test_model(model_path, map_path, model)


def main():
    test()

if __name__ == '__main__':
    main()