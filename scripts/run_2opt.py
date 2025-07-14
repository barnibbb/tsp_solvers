import json
import sys
import time
import math
import os

def greedyPath(cost_matrix):

    # Number of nodes
    n = len(cost_matrix)
    # Keep track of visited nodes
    visited = [False] * n
    # Order of visited nodes
    path = []
    # Total cost of the path
    total_cost = 0

    # Start from first node
    current_node = 0
    visited[current_node] = True
    path.append(current_node)

    for _ in range(n - 1):

        # Find the nearest unvisited neighbor

        min_cost = float('inf')
        next_node = None

        for neighbor in range(n):
            if not visited[neighbor] and cost_matrix[current_node][neighbor] < min_cost:
                min_cost = cost_matrix[current_node][neighbor]
                next_node = neighbor

        if next_node is not None:
            total_cost += min_cost
            visited[next_node] = True
            path.append(next_node)
            current_node = next_node

    total_cost += cost_matrix[current_node][path[0]]

    path.append(path[0])

    return total_cost, path 



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


def compute_tour_cost(path, cost_matrix):
    
    return sum(cost_matrix[path[i]][path[i+1]] for i in range(len(path) - 1)) + cost_matrix[path[-1]][path[0]]




def run_2opt(cost_matrix, initial_path):

    n = len(cost_matrix)
    path = initial_path[:]
    cost = compute_tour_cost(path, cost_matrix)
    improved = True

    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                new_cost = compute_tour_cost(new_path, cost_matrix)

                if new_cost < cost:
                    path = new_path
                    cost = new_cost
                    improved = True


    return cost, path



def read_tsp_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    coordinates = []
    reading_nodes = False

    for line in lines:
        line = line.strip()

        if line == "NODE_COORD_SECTION":
            reading_nodes = True
            continue
        if reading_nodes:
            if line == "EOF":
                break
            parts = line.split()
            coordinates.append((float(parts[1]), float(parts[2])))
        
    return coordinates


def euclidean_2d(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)


def create_cost_matrix(coordinates):
    n = len(coordinates)
    cost_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                cost_matrix[i][j] = euclidean_2d(coordinates[i], coordinates[j])

    return cost_matrix



def read_explicit_tsp_lower_diag(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    cost_matrix = []
    reading_matrix = False

    for line in lines:
        line = line.strip()
        if line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1].strip())
        if line == "EDGE_WEIGHT_SECTION":
            reading_matrix = True
            continue
        if reading_matrix:
            if line == "EOF":
                break
            cost_matrix.extend(map(float, line.split()))

    full_matrix = [[0.0] * dimension for _ in range(dimension)]
    index = 0
    
    # TODO: check dimensions
    for i in range(dimension):
        for j in range(i + 1):
            full_matrix[i][j] = cost_matrix[index]
            full_matrix[j][i] = cost_matrix[index]
            index += 1


    return full_matrix


# TODO: check original formula
def haversine(lat1, lon1, lat2, lon2):
    #R = 6371
    R = 6433.239404800604
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c



def read_geo_tsp(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    coordinates = []
    reading_nodes = False

    for line in lines:
        line = line.strip()

        if line == "NODE_COORD_SECTION":
            reading_nodes = True
            continue
        if reading_nodes:
            if line == "EOF":
                break
            parts = line.split()
            _, lat, lon = parts

            coordinates.append((float(lat), float(lon)))
        
    n = len(coordinates)

    cost_matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                lat1, lon1 = coordinates[i]
                lat2, lon2 = coordinates[j]
                cost_matrix[i][j] = haversine(lat1, lon1, lat2, lon2)

    
    return cost_matrix


def matrix2json(cost_matrix):
    graph = []

    n = len(cost_matrix)

    for i in range(n):
        for j in range(i + 1, n):
            graph.append({
                "distance": cost_matrix[i][j],
                "first_node": i,
                "second_node": j
            })

    json_data = {"graph": graph}

    return json.dumps(json_data, indent=4)



if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python3 run_2opt.py <tsp_file>")
        sys.exit(1)

    tsp_file = sys.argv[1]

    cost_matrix = create_cost_matrix(read_tsp_file(tsp_file))

    # json_output = matrix2json(cost_matrix)

    # output_file = os.path.dirname(tsp_file) + "/graph.json"

    # with open(output_file, "w") as file:
    #     file.write(json_output)

    start_time = time.time()

    greedy_cost, greedy_path = greedyPath(cost_matrix)

    print(f"Greedy min cost: {greedy_cost}")
    # print(f"Greedy path: {greedy_path}")

    opt_cost, opt_path = run_2opt(cost_matrix, greedy_path)

    end_time = time.time()

    execution_time = end_time - start_time

    # print(f"2-opt execution time: {execution_time}.")
    print(f"2-opt min cost: {opt_cost}")
    # print(f"2-opt path: {opt_path}")

