import numpy as np

def load_map(filename):
    return np.load(filename)

def travelling_salesman_problem(cities):
    n_cities = cities.shape[0]
    if n_cities < 2:
        raise ValueError('Number of cities must be greater than 1')
    best_path = None
    best_distance = np.inf
    for i in range(n_cities):
        path = [i]
        distance = 0
        current_city = i
        unvisited_cities = set(range(n_cities))
        unvisited_cities.remove(current_city)
        while unvisited_cities:
            next_city = min(unvisited_cities, key=lambda x: cities[current_city, x])
            path.append(next_city)
            distance += cities[current_city, next_city]
            unvisited_cities.remove(next_city)
            current_city = next_city
        distance += cities[current_city, i]
        path.append(i)
        if distance < best_distance:
            best_path = path
            best_distance = distance
    return best_path, best_distance

def main():
    filename = 'travelling_salesman_problem/maps/map_100_3e70ea85-68b9-4574-bfa5-ba6460da49b6.npy'
    cities = load_map(filename)
    path, distance = travelling_salesman_problem(cities)
    print(f'Best path: {path}')
    print(f'Distance: {distance}')

if __name__ == '__main__':
    main()