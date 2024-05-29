import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import time

def load_map(filename):
    return np.load(filename)

def held_karp(cities):
    n_cities = cities.shape[0]
    
    C = {}
    for k in range(1, n_cities):
        C[(1 << k, k)] = (cities[0, k], 0)
    
    for subset_size in range(2, n_cities):
        for subset in combinations(range(1, n_cities), subset_size):
            bits = sum([1 << bit for bit in subset])
            for k in subset:
                prev_bits = bits & ~(1 << k)
                res = []
                for m in subset:
                    if m == k:
                        continue
                    res.append((C[(prev_bits, m)][0] + cities[m, k], m))
                C[(bits, k)] = min(res)
    
    bits = (2**n_cities - 1) - 1
    res = []
    for k in range(1, n_cities):
        res.append((C[(bits, k)][0] + cities[k, 0], k))
    opt, parent = min(res)
    
    path = [0]
    bits = (2**n_cities - 1) - 1
    for i in range(n_cities - 1):
        path.append(parent)
        bits, parent = bits & ~(1 << parent), C[(bits, parent)][1]
    path.append(0)
    
    path.append(opt)
    
    return np.array(path)


def plot_path(cities, path):
    distance = path[-1]  # Extract the distance
    path = path[:-1].astype(int)  # Convert the rest of the path to integers
    
    sym_cities = (cities + cities.T) / 2
    
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coordinates = mds.fit_transform(sym_cities)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c='red', marker='o')
    
    for i, coord in enumerate(coordinates):
        plt.text(coord[0], coord[1], str(i), fontsize=12, ha='right')

    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            start = coordinates[i]
            end = coordinates[j]
            plt.plot([start[0], end[0]], [start[1], end[1]], 'gray', linestyle='dotted')

    for i in range(len(path) - 1):  # Plot the path excluding the distance
        start = coordinates[path[i]]
        end = coordinates[path[i + 1]]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'b-')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Travelling Salesman Problem - Map and Optimal Path\nTotal Distance: {distance}')
    plt.grid(True)
    plt.show()

def main():

    filename = 'travelling_salesman_problem/maps/map_18_e5b766e4-69d2-4c3e-a952-58d2b248dfa6.npy'
    cities = load_map(filename)
    
    start_time = time.time()
    result = held_karp(cities)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    path, distance = result[:-1], result[-1]
    
    print(f'Best path: {path}')
    print(f'Distance: {distance}')
    print(f'Time taken: {elapsed_time:.2f} seconds')
    
    plot_path(cities, result)

if __name__ == '__main__':
    main()
