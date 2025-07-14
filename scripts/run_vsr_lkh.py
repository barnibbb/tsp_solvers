import sys
import os
import subprocess

def generate_par_file(tsp_file, runs):
    tsp_path = os.path.abspath(tsp_file)
    tsp_dir = os.path.dirname(tsp_path)
    tsp_name = os.path.splitext(os.path.basename(tsp_path))[0]

    tour_path = os.path.join(tsp_dir, tsp_name + ".tour")
    par_path = os.path.join(tsp_dir, tsp_name + ".par")

    with open(par_path, 'w') as f:
        f.write(f"PROBLEM_FILE = {tsp_file}\n")
        f.write(f"OUTPUT_TOUR_FILE = {tour_path}\n")
        f.write(f"RUNS = {runs}\n")

    return par_path
    

def run_vsr_lkh(par_file):
    subprocess.run(["/home/appuser/VSR-LKH/LKH", par_file])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_vsr_lkh.py <tsp_file>")
        sys.exit(1)

    tsp_file = sys.argv[1]
    runs = 1

    par_file = generate_par_file(tsp_file, runs)

    run_vsr_lkh(par_file)


