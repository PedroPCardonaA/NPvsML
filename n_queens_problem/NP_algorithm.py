import numpy as np
import time

def load_map(filename):
    return np.load(filename)

def is_safe(board, row, col):
    # Check the row on left side
    for i in range(col):
        if board[row, i] == 1:
            return False

    # Check the upper diagonal on left side
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i, j] == 1:
            return False

    # Check the lower diagonal on left side
    for i, j in zip(range(row, len(board)), range(col, -1, -1)):
        if board[i, j] == 1:
            return False

    return True

def solve_nqueens_util(board, col):
    if col >= len(board):
        return True

    for i in range(len(board)):
        if is_safe(board, i, col):
            board[i, col] = 1
            if solve_nqueens_util(board, col + 1):
                return True
            board[i, col] = 0  # backtrack

    return False

def solve_nqueens(board):
    if solve_nqueens_util(board, 0):
        return board
    else:
        print("No solution found")
        return board

def main():
    board = load_map('n_queens_problem/boards/board_25_710946b6-a590-40c0-ad0c-d2815ec169fb.npy')
    time_start = time.time()
    result = solve_nqueens(board)
    time_end = time.time()
    print(result)
    print(f"Time taken: {time_end - time_start:.4f} seconds")

if __name__ == '__main__':
    main()
