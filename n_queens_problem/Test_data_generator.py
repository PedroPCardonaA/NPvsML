import numpy as np
import uuid
from Map_generator import generate_random_map, save_board
from NP_algorithm import  solve_nqueens
from Path_generator import get_path

def main():
    path = get_path()
    n_boards = 10
    board_size = 25
    for i in range(n_boards):
        board = generate_random_map(board_size)
        random = np.random.rand()
        board_name = f'board_{board_size}_{uuid.uuid4()}.npy'
        best_board = solve_nqueens(board)
        if random >= 0.9:
            save_board(board, f'{path}/boards/X-data/test/{board_name}')    
            save_board(best_board, f'{path}/boards/Y-data/test/{board_name}')
        elif random >= 0.7:
            save_board(board, f'{path}/boards/X-data/val/{board_name}')
            save_board(best_board, f'{path}/boards/Y-data/val/{board_name}')
        else:
            save_board(board, f'{path}/boards/X-data/train/{board_name}')
            save_board(best_board, f'{path}/boards/Y-data/train/{board_name}')

if __name__ == '__main__':
    main()