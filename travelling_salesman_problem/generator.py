import numpy as np
import uuid

def generate_random_map(n_cities):
    if n_cities < 2:
        raise ValueError('Number of cities must be greater than 1')
    cities = np.random.rand(n_cities, n_cities)
    return cities

def save_map(cities, filename):
    np.save(filename, cities)

def main():
    n_maps = 10
    n_cities = 10
    for i in range(n_maps):
        cities = generate_random_map(n_cities)
        save_map(cities, f'travelling_salesman_problem/maps/map_{n_cities}_{uuid.uuid4()}.npy')

if __name__ == '__main__':
    main()