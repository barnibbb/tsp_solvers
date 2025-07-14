import sys
import os
import time
import math

from concorde.tsp import TSPSolver

def run_concorde(tsp_file):
    tsp_path = os.path.abspath(tsp_file)
    tsp_dir = os.path.dirname(tsp_path)
    tsp_name = os.path.splitext(os.path.basename(tsp_path))[0]
    out_file = os.path.join(tsp_dir, tsp_name + ".sol")

    solver = TSPSolver.from_tspfile(tsp_path)
    # solution = solver.solve(time_bound=60)
    solution = solver.solve()

    with open(out_file, 'w') as f:
        f.write(" ".join(map(str, solution.tour)))


def read_tsplib_coords(file_path):
    coords = {}
    reading = False

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("NODE_COORD_SECTION"):
                reading = True
                continue
            if reading:
                if line == "EOF" or line == "":
                    break
                parts = line.split()
                if len(parts) >= 3:
                    idx = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coords[idx] = (x, y)
    
    return coords


def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compute_cost(coords, tour):
    cost = 0.0
    for i in range(len(tour) - 1):
        cost += euclidean(coords[tour[i]], coords[tour[i+1]])
    cost += euclidean(coords[tour[-1]], coords[tour[0]])

    return cost




if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_concorde.py <tsp_file>")
        sys.exit(1)

    tsp_file = sys.argv[1]

    start = time.time()

    run_concorde(tsp_file)

    stop = time.time()

    print(f"Duration: {stop - start}")

    # tsp_path = os.path.abspath(tsp_file)
    # tsp_dir = os.path.dirname(tsp_path)
    # tsp_name = os.path.splitext(os.path.basename(tsp_path))[0]
    # out_file = os.path.join(tsp_dir, tsp_name + ".sol")

    # with open(out_file, 'r') as f:
    #     tour_str = f.read()

    # tour = [int(i) + 1 for i in tour_str.strip().split() if i.strip() != ""]

    # coords = read_tsplib_coords(tsp_file)

    # cost = compute_cost(coords, tour)

    # print(f"Approximated sol: {cost}")



