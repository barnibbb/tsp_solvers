import sys
import os
import math
import numpy as np
import time

def tsp2matrix(tsp_file):
    with open(tsp_file, "r") as f:
        lines = f.readlines()

    coords = []
    reading_coords = False

    for line in lines:
        if "DIMENSION" in line:
            num_nodes = int(line.split(":")[1].strip())
        if "NODE_COORD_SECTION" in line:
            reading_coords = True
            continue
        elif "EOF" in line:
            break
        if reading_coords:
            parts = line.strip().split()
            idx = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            coords.append((x,y))

    cost_matrix = [[float('inf')] * num_nodes for _ in range(num_nodes)]
    
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            x1, y1 = coords[i]
            x2, y2 = coords[j]
            distance = math.hypot(x2 - x1, y2 - y1)

            cost_matrix[i][j] = distance
            cost_matrix[j][i] = distance
    
    for i in range(num_nodes):
        cost_matrix[i][i] = 0



    return cost_matrix




def json2matrix(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)

    graph = data['graph']

    max_node = 0

    for edge in graph:
        max_node = max(max_node, edge['first_node'], edge['second_node'])

    num_nodes = max_node + 1

    cost_matrix = [[float('inf')] * num_nodes for _ in range(num_nodes)]

    for edge in graph:
        distance = edge['distance']
        first_node = edge['first_node']
        second_node = edge['second_node']

        cost_matrix[first_node][second_node] = distance
        cost_matrix[second_node][first_node] = distance
    
    for i in range(num_nodes):
        cost_matrix[i][i] = 0

    return cost_matrix


def compute_fitness(route, cost_matrix):
    total_cost = 0

    for i in range(len(route) - 1):
        total_cost += cost_matrix[route[i]][route[i + 1]]

    total_cost += cost_matrix[route[-1]][route[0]]

    return total_cost



def greedy_search(cost_matrix, start=0):
    n = len(cost_matrix)
    visited = [False] * n
    tour = [start]
    visited[start] = True
    current = start

    for _ in range(n-1):
        next_city = None
        min_cost = float('inf')

        for city in range(n):
            if not visited[city] and cost_matrix[current][city] < min_cost:
                min_cost = cost_matrix[current][city]
                next_city = city

        if next_city is None:
            raise ValueError(f"No path from node {current} to any unvisited node.")

        tour.append(next_city)
        visited[next_city] = True
        current = next_city

    tour.append(start)

    total_cost = compute_fitness(tour, cost_matrix)

    return tour, total_cost




if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python3 run_greedy.py <tsp_file>")
        sys.exit(1)

    tsp_file = sys.argv[1]

    extension = os.path.splitext(tsp_file)[1]

    if extension == ".json":
        cost_matrix = json2matrix(tsp_file)
    elif extension == ".tsp":
        cost_matrix = tsp2matrix(tsp_file)
    else:
        print("Invalid file format.")
        sys.exit(1)

    start = time.time()

    tour, total_cost = greedy_search(cost_matrix)

    stop = time.time()

    print(f"Length: {len(tour)-1}")

    print(f"Duration: {stop - start}")

    print(f"Total cost: {total_cost}")


