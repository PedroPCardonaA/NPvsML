import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

def load_map(filename):
    return np.load(filename)

def held_karp(cities):
    n_cities = cities.shape[0]
    all_indices = range(n_cities)
    
    # Create a dictionary to store the minimum cost to reach each subset ending in each city
    C = {}
    for k in range(1, n_cities):
        C[(1 << k, k)] = (cities[0, k], 0)
    
    # Iterate subsets of increasing length
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
    
    # Calculate the minimum cost to return to the start city
    bits = (2**n_cities - 1) - 1
    res = []
    for k in range(1, n_cities):
        res.append((C[(bits, k)][0] + cities[k, 0], k))
    opt, parent = min(res)
    
    # Reconstruct the optimal path
    path = [0]
    bits = (2**n_cities - 1) - 1
    for i in range(n_cities - 1):
        path.append(parent)
        bits, parent = bits & ~(1 << parent), C[(bits, parent)][1]
    path.append(0)
    
    return path, opt

def plot_path(cities, path):
    # Symmetrize the cities distance matrix
    sym_cities = (cities + cities.T) / 2
    
    # Use MDS to find a 2D representation of the cities based on the distance matrix
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coordinates = mds.fit_transform(sym_cities)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c='red', marker='o')
    
    for i, coord in enumerate(coordinates):
        plt.text(coord[0], coord[1], str(i), fontsize=12, ha='right')

    # Plot lines between all pairs of cities
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            start = coordinates[i]
            end = coordinates[j]
            plt.plot([start[0], end[0]], [start[1], end[1]], 'gray', linestyle='dotted')

    # Highlight the optimal path
    for i in range(len(path) - 1):
        start = coordinates[path[i]]
        end = coordinates[path[i + 1]]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'b-')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Travelling Salesman Problem - Map and Optimal Path')
    plt.grid(True)
    plt.show()

def main():
    filename = 'travelling_salesman_problem/maps/map_22_f991b792-8782-4073-b869-9d92d948e5db.npy'
    cities = load_map(filename)
    path, distance = held_karp(cities)
    print(f'Best path: {path}')
    print(f'Distance: {distance}')
    plot_path(cities, path)

if __name__ == '__main__':
    main()
