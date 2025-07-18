import sys
import os
import subprocess

def generate_par_file(tsp_file, runs, time_limit):
    tsp_path = os.path.abspath(tsp_file)
    tsp_dir = os.path.dirname(tsp_path)
    tsp_name = os.path.splitext(os.path.basename(tsp_path))[0]

    tour_path = os.path.join(tsp_dir, tsp_name + ".tour")
    par_path = os.path.join(tsp_dir, tsp_name + ".par")

    with open(par_path, 'w') as f:
        f.write(f"PROBLEM_FILE = {tsp_file}\n")
        f.write(f"OUTPUT_TOUR_FILE = {tour_path}\n")
        f.write(f"RUNS = {runs}\n")
        # f.write(f"TIME_LIMIT = {time_limit}\n")

    return par_path
    

def run_lkh(par_file):
    subprocess.run(["LKH", par_file])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_lkh.py <tsp_file>")
        sys.exit(1)

    tsp_file = sys.argv[1]
    runs = 1
    time_limit = 600.0

    par_file = generate_par_file(tsp_file, runs, time_limit)

    run_lkh(par_file)
