# IF path /travelling_salesman_problem/ exists, return it. Otherwise, return /content/NPvsML/travelling_salesman_problem/
import os

def get_path():
    path = 'n_queens_problem/'
    if os.path.exists(path):
        return path
    else:
        return '/content/NPvsML/n_queens_problem/'