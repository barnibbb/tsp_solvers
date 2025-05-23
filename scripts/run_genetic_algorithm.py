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


def tournament_selection(population, cost_matrix, k=5):
    tournament = random.sample(population, k)

    return min(tournament, key=lambda ind: compute_fitness(ind, cost_matrix))



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
    mutant = route.copy()
    size = len(mutant)
    point1, point2 = sorted(random.sample(range(size), 2))
    # Inclusive interval
    mutant[point1:point2 + 1] = reversed(mutant[point1:point2 + 1])

    return mutant


def swap_mutation(route):
    mutant = route.copy()
    a, b = random.sample(range(len(mutant)), 2)
    mutant[a], mutant[b] = mutant[b], mutant[a]

    return mutant


def scramble_mutation(route):
    mutant = route.copy()
    a, b = sorted(random.sample(range(len(mutant)), 2))
    subsequence = mutant[a:b+1]
    random.shuffle(subsequence)

    return mutant[:a] + subsequence + mutant[b+1:]


def insert_mutation(route):
    mutant = route.copy()
    a, b = random.sample(range(len(mutant)), 2)
    node = mutant.pop(a)
    mutant.insert(b, node)

    return mutant



def truncation_selection(population, cost_matrix, num_survivors):
    fitness_cache = {}
    normalized_counts = {}
    penalized_fitness = []

    for ind in population:
        ind = normalize_tour(ind)

        norm = tuple(ind)

        # Count appearance
        normalized_counts[norm] = normalized_counts.get(norm, 0) + 1

        # Cache raw fitness
        if norm not in fitness_cache:
            fitness_cache[norm] = compute_fitness(ind, cost_matrix)
        
        penalty = (normalized_counts[norm] - 1) * 30
        penalized_fitness.append((fitness_cache[norm] + penalty, ind))

    penalized_fitness.sort(key=lambda x: x[0])

    selected = penalized_fitness[:num_survivors]
    selected_inds = [normalize_tour(ind) for (_, ind) in selected]
    selected_fitness = [mod_fitness for (mod_fitness, _) in selected]

    return selected_inds, selected_fitness



def normalize_tour(tour):
    idx = tour.index(min(tour))

    return tour[idx:] + tour[:idx]




def genetic_algorithm(cost_matrix, population_size = 150, num_generations = 200, num_survivors = 75, selection_type="fp", mutation_type="inversion"):

    n = len(cost_matrix)

    # Generating initial population
    population = [random.sample(range(n), n) for _ in range(population_size)] 

    for generation in range(num_generations):
        offsprings = []
        mutants = []

        # Peform PMX crossover to obtain offsprings
        for _ in range(population_size // 2):

            if selection_type == "fp":
                parent1 = fp_selection(population, cost_matrix)
                parent2 = fp_selection(population, cost_matrix)
            elif selection_type == "tournament":                
                parent1 = tournament_selection(population, cost_matrix)
                parent2 = tournament_selection(population, cost_matrix)
            else:
                print("Selection type must be set")

            child1, child2 = pmx_crossover(parent1, parent2)

            if child1 != parent1 and child1 != parent2:
                offsprings.append(child1)

            if child2 != parent1 and child2 != parent2:
                offsprings.append(child2)


        # Perform mutation in 30% of the offsprings    
        for individual in offsprings:
            if random.random() < 0.3:
                if mutation_type == "inversion":
                    mutant = inversion(individual)
                elif mutation_type == "swap":
                    mutant = swap_mutation(individual)
                elif mutation_type == "scramble":
                    mutant = scramble_mutation(individual)
                elif mutation_type == "insert":
                    mutant = insert_mutation(individual)
                else:
                    print("Mutation type must be set")
                mutants.append(mutant)
                # print(individual)
                # print(mutant)
                # return

        # Select survivals for the next generation with truncation
        population, fitnesses = truncation_selection(population + offsprings + mutants, cost_matrix, num_survivors)

        best_individual = min(population, key=lambda ind: compute_fitness(ind, cost_matrix))

        # Print best individual in each iteration
        # print(f"Generation {generation}, Best fitness: {compute_fitness(best_individual, cost_matrix)}, Population size: {len(population + offsprings + mutants)}")        

    return best_individual




if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Usage: python3 run_genetic_algorithm.py <tsp_file> <selection_type> <mutation_type>")
        sys.exit(1)

    tsp_file = sys.argv[1]
    selection_type = sys.argv[2]
    mutation_type = sys.argv[3]

    extension = os.path.splitext(tsp_file)[1]

    if extension == ".json":
        cost_matrix = json2matrix(tsp_file)
    elif extension == ".tsp":
        cost_matrix = tsp2matrix(tsp_file)
    else:
        print("Invalid file format.")
        sys.exit(1)


    for _ in range(0, 10):
        start_time = time.time()

        best_route = genetic_algorithm(cost_matrix, population_size = 200, num_generations = 500, num_survivors = 100, selection_type=selection_type, mutation_type=mutation_type)

        end_time = time.time()

        execution_time = end_time - start_time

        print(f"{compute_fitness(best_route, cost_matrix)} {execution_time}")

    # print(f"Best route: {best_route}")
    # print(f"Total cost: {compute_fitness(best_route, cost_matrix)}")
    # print(f"GA execution time: {execution_time}.")
