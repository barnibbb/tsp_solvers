import mlrose_hiive
import sys

def parse_tsplib(filename):
    coords = []
    reading_nodes = False
    with open(filename, 'r') as f:
        for line in f:
            if 'NODE_COORD_SECTION' in line:
                reading_nodes = True
                continue
            if 'EOF' in line:
                break
            if reading_nodes:
                parts = line.strip().split()
                if len(parts) >= 3:
                    x, y = float(parts[1]), float(parts[2])
                    coords.append((x, y))
    return coords



if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python3 run_mlrose.py <tsp_file>")
        sys.exit(1)

    tsp_file = sys.argv[1]

    coords = parse_tsplib(tsp_file)

    fitness = mlrose_hiive.TravellingSales(coords=coords)
    problem = mlrose_hiive.TSPOpt(length=len(coords), fitness_fn=fitness, maximize=False)

    for i in range(10):
        best_state, best_fitness, _ = mlrose_hiive.genetic_alg(
            problem,
            pop_size=500,
            mutation_prob=0.3,
            max_attempts=500,
            max_iters=1000,
            random_state=None,
            curve=True
        )

        print(best_fitness)

