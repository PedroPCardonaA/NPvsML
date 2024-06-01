import numpy as np
import uuid

def generate_random_map(board_size):
    if board_size < 1:
        raise ValueError('Board size must be greater than 0')
    
    board = np.zeros((board_size, board_size))
    num_queens = 0
    
    for _ in range(num_queens):
        placed = False
        while not placed:
            row = np.random.randint(0, board_size)
            col = np.random.randint(0, board_size)
            
            if board[row, col] == 0:  # Check if the position is empty
                # Check if the queen is attacking any other queen
                if not any(board[row, :] == 1) and not any(board[:, col] == 1):
                    if not any(np.diag(board, col - row) == 1) and not any(np.diag(np.fliplr(board), board_size - 1 - col - row) == 1):
                        board[row, col] = 1
                        placed = True
    
    return board

def save_board(board, filename):
    np.save(filename, board)

def main():
    n_boards = 1
    board_size = 25
    for i in range(n_boards):
        board = generate_random_map(board_size)
        print(board)
        save_board(board, f'n_queens_problem/boards/board_{board_size}_{uuid.uuid4()}.npy')

if __name__ == '__main__':
    main()