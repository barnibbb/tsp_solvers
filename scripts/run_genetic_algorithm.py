import json
import random
import sys
import time
import os
import math
import time
import concurrent.futures
from collections import defaultdict


## Conversion functions
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
### Conversion functions




## Fitness computation
def compute_fitness(route, cost_matrix):
    total_cost = 0

    for i in range(len(route) - 1):
        total_cost += cost_matrix[route[i]][route[i + 1]]

    total_cost += cost_matrix[route[-1]][route[0]]

    return total_cost


def normalize_tour(tour):
    idx = tour.index(min(tour))

    return tour[idx:] + tour[:idx]
### Fitness computation




## Selections
def fp_selection(population, cost_matrix):
    fitnesses = [compute_fitness(individual, cost_matrix) for individual in population]

    total_fitness = sum(fitnesses)

    probabilities = [1.0 - (f / total_fitness) for f in fitnesses]

    selected = random.choices(population, weights=probabilities, k=1)

    return selected[0]


def tournament_selection(population, cost_matrix, k=5):
    tournament = random.sample(population, k)

    return min(tournament, key=lambda ind: compute_fitness(ind, cost_matrix))


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


def truncation_selection_2(population, cost_matrix, num_survivors, precomputed_fitness=None):
    if precomputed_fitness is None:
        fitnesses = [compute_fitness(ind, cost_matrix) for ind in population]
    else:
        fitness = precomputed_fitness

    sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1])
    selected = [ind for ind, _ in sorted_population[:num_survivors]]
    selected_fitness = [fit for _, fit in sorted_population[:num_survivors]]

    return selected, selected_fitness



def select(population, cost_matrix, selection_type):
    if selection_type == "fp":
        return fp_selection(population, cost_matrix)
    elif selection_type == "tournament":                
        return tournament_selection(population, cost_matrix)
    else:
        raise ValueError("Unknown selection type")
### Selections




## Crossovers
def pmx_crossover(parent1, parent2):
    size = len(parent1)
    point1, point2 = sorted(random.sample(range(size), 2))

    def create_pmx_child(p1, p2):
        child = [None] * size
        child[point1:point2 + 1] = p1[point1:point2 + 1]

        mapping = {p1[i]: p2[i] for i in range(point1, point2 + 1)}

        for i in range(size):
            if child[i] is None:
                value = p2[i]
                while value in mapping:
                    value = mapping[value]
                child[i] = value

        return child

    child1 = create_pmx_child(parent1, parent2)
    child2 = create_pmx_child(parent2, parent1)

    return child1, child2


def csrx_pmx(parent1, parent2):
    size = len(parent1)
    point1, point2 = sorted(random.sample(range(size), 2))

    def create_pmx_child(p1, p2):
        child = [None] * size
        child[point1:point2 + 1] = p1[point1:point2 + 1]

        mapping = {p1[i]: p2[i] for i in range(point1, point2 + 1)}

        for i in range(size):
            if child[i] is None:
                value = p2[i]
                while value in mapping:
                    value = mapping[value]
                child[i] = value

        return child

    target = parent1[point1]
    shifted = circular_shift(parent2, target, point1)
    rshifted = circular_rshift(parent2, target, point1)

    child1 = create_pmx_child(parent1, shifted)
    child2 = create_pmx_child(parent1, rshifted)

    cost1 = compute_fitness(child1, cost_matrix)
    cost2 = compute_fitness(child2, cost_matrix)

    return child1 if cost1 < cost2 else child2



def ox_crossover(parent1, parent2):
    size = len(parent1)
    point1, point2 = sorted(random.sample(range(size), 2))

    def create_ox_child(p1, p2):
        child = [None] * size
        child[point1:point2 + 1] = p1[point1:point2 + 1]

        p2_index = (point2 + 1) % size
        c_index = (point2 + 1) % size

        while None in child:
            gene = p2[p2_index]
            if gene not in child:
                child[c_index] = gene
                c_index = (c_index + 1) % size
            p2_index = (p2_index + 1) % size

        return child

    child1 = create_ox_child(parent1, parent2)
    child2 = create_ox_child(parent2, parent1)

    return child1, child2


def csrx_ox(parent1, parent2):
    size = len(parent1)
    point1, point2 = sorted(random.sample(range(size), 2))

    def create_ox_child(p1, p2):
        child = [None] * size
        child[point1:point2 + 1] = p1[point1:point2 + 1]

        p2_index = (point2 + 1) % size
        c_index = (point2 + 1) % size

        while None in child:
            gene = p2[p2_index]
            if gene not in child:
                child[c_index] = gene
                c_index = (c_index + 1) % size
            p2_index = (p2_index + 1) % size

        return child

    target = parent1[point1]
    shifted = circular_shift(parent2, target, point1)
    rshifted = circular_rshift(parent2, target, point1)

    child1 = create_ox_child(parent1, shifted)
    child2 = create_ox_child(parent1, rshifted)

    cost1 = compute_fitness(child1, cost_matrix)
    cost2 = compute_fitness(child2, cost_matrix)

    return child1 if cost1 < cost2 else child2




def box_crossover(parent1, parent2):
    size = len(parent1)
    point1, point2 = sorted(random.sample(range(size), 2))

    def create_box_child(p1, p2):
        child = [None] * size
        child[point1:point2+1] = parent1[point1:point2+1]

        remaining_genes = [gene for gene in parent2 if gene not in child[point1:point2+1]] 

        random.shuffle(remaining_genes)

        # Fill the rest of child
        idx = 0
        for k in range(size):
            if child[k] is None:
                child[k] = remaining_genes[idx]
                idx += 1
            
        return child

    child1 = create_box_child(parent1, parent2)
    child2 = create_box_child(parent2, parent1)

    return child1, child2




def erx_crossover(parent1, parent2):
    edge_map = build_edge_map(parent1, parent2)
    size = len(parent1)

    def create_erx_child(p1, p2):
        current = random.choice(parent1)
        child = [current]

        while len(child) < size:
            # Remove current from all edge lists
            for edges in edge_map.values():
                edges.discard(current)

            if edge_map[current]:
                # Choose the neighbor with fewest entries in edge map (degree)
                next_city = min(edge_map[current], key=lambda x: len(edge_map[x]))
            else:
                # Pick a random remaining city
                remaining = list(set(parent1) - set(child))
                next_city = random.choice(remaining)

            child.append(next_city)
            current = next_city

        return child


    child1 = create_erx_child(parent1, parent2)
    child2 = create_erx_child(parent2, parent1)

    return child1, child2


def build_edge_map(parent1, parent2):
    edge_map = defaultdict(set)
    parents = [parent1, parent2]

    for parent in parents:
        for i in range(len(parent)):
            left = parent[i - 1]
            right = parent[(i + 1) % len(parent)]
            edge_map[parent[i]].update([left, right])
    
    return edge_map


def circular_shift(parent, target_value, index):
    # Shift target value to index
    pos = parent.index(target_value)
    shift = (index - pos) % len(parent)
    return parent[shift:] + parent[:shift]


def circular_rshift(parent, target_value, index):
    pos = parent.index(target_value)
    shift = (pos - index) % len(parent)
    return parent[-shift:] + parent[:-shift]



def cross(parent1, parent2, crossover_type):
    if crossover_type == "pmx":
        return pmx_crossover(parent1, parent2)
    elif crossover_type == "ox":
        return ox_crossover(parent1, parent2)
    elif crossover_type == "box":
        return box_crossover(parent1, parent2)
    elif crossover_type == "erx":
        return erx_crossover(parent1, parent2)
    else:
        raise ValueError("Unknown crossover type")


def csrx_cross(parent1, parent2, crossover_type):
    if crossover_type == "csrx_pmx":
        return csrx_pmx(parent1, parent2)
    elif crossover_type == "csrx_ox":
        return csrx_ox(parent1, parent2)
    else:
        raise ValueError("Unknown crossover type")
### Crossovers






## Mutations
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


def mutate(individual, mutation_type):
    if mutation_type == "inversion":
        return inversion(individual)
    elif mutation_type == "swap":
        return swap_mutation(individual)
    elif mutation_type == "scramble":
        return scramble_mutation(individual)
    elif mutation_type == "insert":
        return insert_mutation(individual)
    else:
        raise ValueError("Unknown mutation type")


def parallel_mutation(offsprings, mutation_func, mutation_rate):
    def mutate(ind):
        if random.random() < mutation_rate:
            return mutation_func(ind)
        return ind

    with concurrent.futures.ThreadPoolExecutor() as executor:
        mutants = list(executor.map(mutate, offsprings))
    
    return mutants
### Mutations





## Elitism
def elitism(population, cost_matrix, percent=0.1):
    sorted_population = sorted(population, key=lambda ind: compute_fitness(ind, cost_matrix))

    num_top = max(1, int(len(population) * percent))

    return sorted_population[:num_top]
### Elitism




## Local search
def simulated_annealing(tour, cost_matrix, T0=100, alpha=0.995, T_min=1e-3, max_iter=100):
    current = tour[:]
    best = tour[:]
    best_cost = current_cost = compute_fitness(current, cost_matrix)
    T = T0

    for _ in range(max_iter):
        if T < T_min:
            break

        i, j = sorted(random.sample(range(len(tour)), 2))
        neighbor = current[:]
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        neighbor_cost = compute_fitness(neighbor, cost_matrix)

        delta = neighbor_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / T):
            current = neighbor[:]
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost

        T *= alpha
    
    return best



def run_2opt(initial_path, cost_matrix):
    n = len(cost_matrix)
    path = initial_path[:]
    cost = compute_fitness(path, cost_matrix)
    improved = True

    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                new_cost = compute_fitness(new_path, cost_matrix)

                if new_cost < cost:
                    path = new_path
                    cost = new_cost
                    improved = True

    return path
### Local search




## Core algorithm
def genetic_algorithm(cost_matrix, population_size = 150, num_generations = 200, num_survivors = 150, num_crossovers=75, selection_type="fp", crossover_type="pmx", mutation_type="inversion"):

    n = len(cost_matrix)

    mutation_rate = 0.3
    percent = 0.1
    sa_rate = 0.2
    opt_rate = 0.05

    # Generating initial population
    population = [random.sample(range(n), n) for _ in range(population_size)]

    crossover_map = {
        "pmx": pmx_crossover,
        "csrx_pmx": csrx_pmx, 
        "ox": ox_crossover,
        "csrx_ox": csrx_ox,
        "box": box_crossover,
        "erx": erx_crossover
    }

    mutation_map = {
        "inversion": inversion,
        "swap": swap_mutation,
        "scramble": scramble_mutation,
        "insert": insert_mutation
    }

    if crossover_type not in crossover_map:
        raise ValueError("Unknown crossover type")

    if mutation_type not in mutation_map:
        raise ValueError("Unknown mutation type")

    crossover_func = crossover_map[crossover_type]

    mutate_func = mutation_map[mutation_type]

    for generation in range(num_generations):
        offsprings = []

        # Peform PMX crossover to obtain offsprings
        for _ in range(num_crossovers):
            parent1 = select(population, cost_matrix, selection_type)    
            parent2 = select(population, cost_matrix, selection_type)
            child1, child2 = cross(parent1, parent2, crossover_type)
            offsprings.extend([child1, child2])
            # child = csrx_cross(parent1, parent2, crossover_type)
            # offsprings.extend([child])

        # Perform mutation - single thread     
        for i in range(len(offsprings)):
            if random.random() < mutation_rate:
                offsprings[i] = mutate(offsprings[i], mutation_type)

        # Perform mutation - multi thread
        # mutants = parallel_mutation(offsprings, mutate_func, mutation_rate=0.3)

        # Elitism
        elite = elitism(population, cost_matrix, percent)

        # Random individuals
        num_inds = 20
        new_inds = [random.sample(range(n), n) for _ in range(num_inds)]

        # candidates = elite + offsprings + new_inds # + mutants
        candidates = offsprings

        # Local search: simulated annealing
        # for i in range(len(candidates)):
        #     if random.random() < sa_rate:
        #         candidates[i] = simulated_annealing(candidates[i], cost_matrix)

        # Local search: 2-opt
        # for i in range(len(candidates)):
        #     if random.random() < opt_rate:
        #         candidates[i] = run_2opt(candidates[i], cost_matrix)

        # Select survivals for the next generation with truncation
        population, fitnesses = truncation_selection_2(candidates, cost_matrix, num_survivors)

        best_individual = min(population, key=lambda ind: compute_fitness(ind, cost_matrix))

        # Print best individual in each iteration
        # print(f"Generation {generation}, Best fitness: {compute_fitness(best_individual, cost_matrix)}, Population size: {len(population)}")        


    return best_individual
### Core algorithm



if __name__ == "__main__":

    if len(sys.argv) < 5:
        print("Usage: python3 run_genetic_algorithm.py <tsp_file> <selection_type> <crossover_type> <mutation_type>")
        sys.exit(1)

    tsp_file = sys.argv[1]
    selection_type = sys.argv[2]
    crossover_type = sys.argv[3]
    mutation_type = sys.argv[4]

    extension = os.path.splitext(tsp_file)[1]

    if extension == ".json":
        cost_matrix = json2matrix(tsp_file)
    elif extension == ".tsp":
        cost_matrix = tsp2matrix(tsp_file)
    else:
        print("Invalid file format.")
        sys.exit(1)


    num_crossovers = 200 if crossover_type == "csrx_pmx" or crossover_type == "csrx_ox" else 100 

    for _ in range(0, 10):
        start_time = time.time()

        best_route = genetic_algorithm(cost_matrix, population_size = 200, num_generations = 500, num_survivors = 200, num_crossovers=num_crossovers, selection_type=selection_type, crossover_type=crossover_type, mutation_type=mutation_type)

        end_time = time.time()

        execution_time = end_time - start_time

        print(f"{compute_fitness(best_route, cost_matrix)} {execution_time}")


