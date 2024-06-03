import torch
from ML_model import NQueensModel, define_model
from NP_algorithm import load_map, plot_solution, solve_nqueens
import time
from Path_generator import get_path

def test_model(model_path, map_path, model: NQueensModel = None):
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
    
    board = model(X)
    board = board.squeeze().detach().cpu().numpy()
    time_end = time.time()
    print(f'Time taken: {time_end - time_start:.2f} seconds')
    
    # Plot path
    plot_solution(board)

def test(path):
    model_path = f'{path}/models/model.pth'
    map_path = f'{path}/boards/board_25_21d6402a-ce56-4fa8-a941-829e2c489bbb.npy'
    board = load_map(map_path)

    print('N_queens_solution Algorithm:')
    
    start_time = time.time()
    result = solve_nqueens(board)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    board = result

    print(f'Time taken: {elapsed_time:.2f} seconds')
    
    plot_solution(board)

    print('Neural Network Model:')
    model = define_model()
    test_model(model_path, map_path, model)

def main():
    path = get_path()
    test(path)

if __name__ == '__main__':
    main()
