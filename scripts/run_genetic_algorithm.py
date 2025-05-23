import json
import random
import sys
import time
import os
import math
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



def fp_selection(population, cost_matrix):
    fitnesses = [compute_fitness(individual, cost_matrix) for individual in population]

    total_fitness = sum(fitnesses)

    probabilities = [1.0 - (f / total_fitness) for f in fitnesses]

    selected = random.choices(population, weights=probabilities, k=1)

    return selected[0]



def pmx_crossover(parent1, parent2):
    size = len(parent1)
    child1, child2 = [-1] * size, [-1] * size

    point1, point2 = sorted(random.sample(range(size), 2))

    # Inclusive interval
    child1[point1:point2 + 1] = parent1[point1:point2 + 1]
    child2[point1:point2 + 1] = parent2[point1:point2 + 1]

    mapping1 = {parent1[i]: parent2[i] for i in range(point1, point2 + 1)}
    mapping2 = {parent2[i]: parent1[i] for i in range(point1, point2 + 1)}

    def fill_missing(child, parent, mapping):
        for i in range(size):
            if child[i] == -1:
                value = parent[i]
                while value in mapping:
                    value = mapping[value]
                child[i] = value
    
    fill_missing(child1, parent2, mapping1)
    fill_missing(child2, parent1, mapping2)

    return child1, child2



def inversion(route):
    size = len(route)

    point1, point2 = sorted(random.sample(range(size), 2))

    print(point1, point2)

    # Inclusive interval
    route[point1:point2 + 1] = reversed(route[point1:point2 + 1])

    return route



def truncation_selection(population, cost_matrix, num_survivors):
    population = sorted(population, key=lambda ind: compute_fitness(ind, cost_matrix))

    return population[:num_survivors]



def genetic_algorithm(cost_matrix, population_size = 150, num_generations = 200, num_survivors = 75):

    n = len(cost_matrix)

    population = [random.sample(range(n), n) for _ in range(population_size)]

    for generation in range(num_generations):
        offspring = []
        mutants = []

        for _ in range(population_size // 2):
            for _ in range(100):
                parent1 = fp_selection(population, cost_matrix)
                parent2 = fp_selection(population, cost_matrix)

                if (parent1 != parent2):
                    break

            child1, child2 = pmx_crossover(parent1, parent2)

            if child1 != parent1 and child1 != parent2:
                offspring.append(child1)

            if child2 != parent1 and child2 != parent2:
                offspring.append(child2)

            
        for individual in offspring:
            if random.random() < 0.3:
                mutant = inversion(individual)
                mutants.append(mutant)

        population = truncation_selection(population + offspring + mutants, cost_matrix, num_survivors)

        best_individual = min(population, key=lambda ind: compute_fitness(ind, cost_matrix))

        print(f"Generation {generation}, Best fitness: {compute_fitness(best_individual, cost_matrix)}, Population size: {len(population + offspring + mutants)}")

    for i in range(len(population)):
        print(i, population[i])
        print('\n')


    return best_individual





if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python3 run_genetic_algorithm.py <tsp_file>")
        sys.exit(1)

    tsp_file = sys.argv[1]

    start_time = time.time()

    extension = os.path.splitext(tsp_file)[1]

    if extension == ".json":
        cost_matrix = json2matrix(tsp_file)
    elif extension == ".tsp":
        cost_matrix = tsp2matrix(tsp_file)
    else:
        print("Invalid file format.")
        sys.exit(1)

    best_route = genetic_algorithm(cost_matrix)

    end_time = time.time()

    execution_time = end_time - start_time

    print(f"Best route: {best_route}")
    print(f"Total cost: {compute_fitness(best_route, cost_matrix)}")
    print(f"GA execution time: {execution_time}.")
