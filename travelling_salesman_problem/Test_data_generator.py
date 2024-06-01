import numpy as np
import uuid
from Map_generator import generate_random_map, save_map
from NP_algorithm import load_map, held_karp, plot_path
from Path_generator import get_path



def main():
    path = get_path()
    n_cities = 18
    n_maps = 1000
    for i in range(n_maps):
        cities = generate_random_map(n_cities)
        random = np.random.rand()
        map_name = f'map_{n_cities}_{uuid.uuid4()}.npy'
        best_path = held_karp(cities)
        if random >= 0.9:
            save_map(cities, f'{path}/maps/X-data/test/{map_name}')    
            save_map(best_path, f'{path}/maps/Y-data/test/{map_name}')
        elif random >= 0.7:
            save_map(cities, f'{path}/maps/X-data/val/{map_name}')
            save_map(best_path, f'{path}/maps/Y-data/val/{map_name}')
        else:
            save_map(cities, f'{path}/maps/X-data/train/{map_name}')
            save_map(best_path, f'{path}/maps/Y-data/train/{map_name}')

if __name__ == '__main__':
    main()