# TSP solvers

## Setup

```bash
./build_docker.sh
./run_docker.sh
```

## Concorde (pyconcorde)

<https://www.math.uwaterloo.ca/tsp/concorde.html>

```bash
python3 run_concorde.py <tsp_file>
```

## LKH

<http://webhotel4.ruc.dk/~keld/research/LKH-3/>

```bash
python3 run_lkh.py <tsp_file>
```

## NeuroLKH

<https://github.com/liangxinedu/NeuroLKH/tree/main>

```bash
python3 run_neurolkh.py --tsplib <bool> --instance_name <str> --optimal_value <int>
```

The optimal values can be an estimate.

## Genetic algorithm

```bash
python3 run_genetic_algorithm.py <tsp_file>
```

The tsp_file can either be .tsp (tsplib) or .json (real).

## Graph generation

Random graphs can be generated in tsplib EUC_2D format.

```bash
python3 generate_graph.py <output_folder> <num_nodes>
```

## TODO

- Check alternatives for GA diversity
