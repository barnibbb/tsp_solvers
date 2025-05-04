import os
from subprocess import check_call
from multiprocessing import Pool
import tqdm
import numpy as np
import pickle

import torch
from torch.autograd import Variable
from tqdm import trange
import argparse
import time
import tempfile

import json
import sys

sys.path.append('/home/appuser/NeuroLKH/')
from net.sgcn_model import SparseGCNModel


def write_para(instance_name, method, para_filename, opt_value, tsplib):
    with open(para_filename, "w") as f:
        if tsplib == True:
            f.write("PROBLEM_FILE = /home/appuser/NeuroLKH/tsplib_data/" + instance_name + ".tsp\n")
        else:
            f.write("PROBLEM_FILE = /home/appuser/data/real/" + instance_name + "/" + instance_name + ".tsp\n")
        f.write("MOVE_TYPE = 5\nPATCHING_C = 3\nPATCHING_A = 2\nRUNS = 100\n")
        f.write("OPTIMUM = " + str(opt_value) + "\n")
        if method == "NeuroLKH":
            f.write("SEED = 1234\n")
            f.write("CANDIDATE_FILE = /home/appuser/NeuroLKH/result/tsplib/candidate/" + instance_name + ".txt\n")
        elif method == "FeatGenerate":
            f.write("GerenatingFeature\n")
            f.write("Feat_FILE = /home/appuser/NeuroLKH/result/" + dataset_name + "/feat/" + instance_name + ".txt\n")
        else:
            assert method == "LKH"

def read_feat(feat_filename):
    edge_index = []
    edge_feat = []
    inverse_edge_index = []
    with open(feat_filename, "r") as f:
        lines = f.readlines()
        for line in lines[:-1]:
            line = line.strip().split()
            for i in range(20):
                edge_index.append(int(line[i * 3]))
                edge_feat.append(int(line[i * 3 + 1]) / 1000000)
                inverse_edge_index.append(int(line[i * 3 + 2]))
    edge_index = np.array(edge_index).reshape(1, -1, 20)
    edge_feat = np.array(edge_feat).reshape(1, -1, 20)
    inverse_edge_index = np.array(inverse_edge_index).reshape(1, -1, 20)
    runtime = float(lines[-1].strip())
    return edge_index, edge_feat, inverse_edge_index, runtime

def write_candidate_pi(dataset_name, instance_name, candidate, pi):
    n_node = candidate.shape[0]
    with open("/home/appuser/NeuroLKH/result/" + dataset_name + "/candidate/" + instance_name + ".txt", "w") as f:
        f.write(str(n_node) + "\n")
        for j in range(n_node):
            line = str(j + 1) + " 0 5"
            for _ in range(5):
                line += " " + str(int(candidate[j, _]) + 1) + " " + str(_ * 100)
            f.write(line + "\n")
        f.write("-1\nEOF\n")
    with open("/home/appuser/NeuroLKH/result/" + dataset_name + "/pi/" + instance_name + ".txt", "w") as f:
        f.write(str(n_node) + "\n")
        for j in range(n_node):
            line = str(j + 1) + " " + str(int(pi[j]))
            f.write(line + "\n")
        f.write("-1\nEOF\n")

def method_wrapper(args):
    if args[0] == "LKH":
        return solve_LKH(*args[1:])
    elif args[0] == "NeuroLKH":
        return solve_NeuroLKH(*args[1:])
    elif args[0] == "FeatGen":
        return generate_feat(*args[1:])

def solve_LKH(instance_name, opt_value):
    para_filename = "/home/appuser/NeuroLKH/result/tsplib/LKH_para/" + instance_name + ".para"
    log_filename = "/home/appuser/NeuroLKH/result/tsplib/LKH_log/" + instance_name + ".log"
    write_para(instance_name, "LKH", para_filename, opt_value)
    with open(log_filename, "w") as f:
        check_call(["./LKH", para_filename], stdout=f)
    return read_results(log_filename)


def infer_SGN_write_candidate2(net, instance_names):
    for instance_name in instance_names:
        # Read coordinates
        with open("/home/appuser/data/real/" + instance_name + "/waypoints.json") as f:
            waypoints_data = json.load(f)

        waypoints = waypoints_data['waypoints']
        n_nodes = len(waypoints)

        coords = np.zeros((n_nodes, 2))

        for wp in waypoints:
            coords[wp['id']] = [wp['x'], wp['y']]

        x = coords.copy()
        x = x.reshape(1, n_nodes, 2)

        # Read distances
        with open("/home/appuser/data/real/" + instance_name + "/graph.json") as f:
            graph_data = json.load(f)

        cost_matrix = np.full((n_nodes, n_nodes), np.inf)

        for edge in graph_data['graph']:
            i = edge['first_node']
            j = edge['second_node']
            d = edge['distance']
            cost_matrix[i, i] = 0
            cost_matrix[j, j] = 0
            cost_matrix[i, j] = d
            cost_matrix[j, i] = d

        # Set input for NN
        n_edges = 20
        batch_size = 1
        node_feat = x

        dist = cost_matrix[np.newaxis, :, :]

        edge_index = np.argsort(dist, axis=-1)[:, :, 1:1+n_edges]
        edge_feat = dist[np.arange(batch_size).reshape(-1, 1, 1), np.arange(n_nodes).reshape(1, -1, 1), edge_index]
        inverse_edge_index = -np.ones(shape=[batch_size, n_nodes, n_nodes], dtype="int")
        inverse_edge_index[np.arange(batch_size).reshape(-1, 1, 1), edge_index, np.arange(n_nodes).reshape(1, -1, 1)] = np.arange(n_edges).reshape(1, 1, -1) + np.arange(n_nodes).reshape(1, -1, 1) * n_edges
        inverse_edge_index = inverse_edge_index[np.arange(batch_size).reshape(-1, 1, 1), np.arange(n_nodes).reshape(1, -1, 1), edge_index]
        edge_index_np = edge_index

        node_feat = Variable(torch.FloatTensor(node_feat).type(torch.cuda.FloatTensor), requires_grad=False) # B x N x 2
        edge_feat = Variable(torch.FloatTensor(edge_feat).type(torch.cuda.FloatTensor), requires_grad=False).view(batch_size, -1, 1) # B x 20N x 1
        edge_index = Variable(torch.LongTensor(edge_index).type(torch.cuda.LongTensor), requires_grad=False).view(batch_size, -1) # B x 20N
        inverse_edge_index = Variable(torch.FloatTensor(inverse_edge_index).type(torch.cuda.LongTensor), requires_grad=False).view(batch_size, -1) # B x 20N
        candidate_test = []
        label = None
        edge_cw = None

        y_edges, _, y_nodes = net.forward(node_feat, edge_feat, edge_index, inverse_edge_index, label, edge_cw, n_edges)
        y_edges = y_edges.detach().cpu().numpy()
        y_edges = y_edges[:, :, 1].reshape(batch_size, n_nodes, n_edges)
        y_edges = np.argsort(-y_edges, -1)
        edge_index = np.array(edge_index_np)
        candidate_index = edge_index[np.arange(batch_size).reshape(-1, 1, 1), np.arange(n_nodes).reshape(1, -1, 1), y_edges]
        candidate_test.append(candidate_index[:, :, :5])
        candidate_test = np.concatenate(candidate_test, 0)
        with open("/home/appuser/NeuroLKH/result/tsplib/candidate/" + instance_name + ".txt", "w") as f:
            f.write(str(n_nodes) + "\n")
            for j in range(n_nodes):
                line = str(j + 1) + " 0 5"
                for _ in range(5):
                    line += " " + str(candidate_test[0, j, _] + 1) + " 1"
                f.write(line + "\n")
            f.write("-1\nEOF\n")
        

        


def infer_SGN_write_candidate(net, instance_names):
    for instance_name in instance_names:
        with open("/home/appuser/NeuroLKH/tsplib_data/" + instance_name + ".tsp", "r") as f:
            lines = f.readlines()
            assert lines[4] == "EDGE_WEIGHT_TYPE : EUC_2D\n" 
            assert lines[5] == "NODE_COORD_SECTION\n"
            n_nodes = int(lines[3].split(" ")[-1])
            x = []
            for i in range(n_nodes):
                line = [float(_) for _ in lines[6 + i].strip().split()]
                assert len(line) == 3
                assert line[0] == i + 1
                x.append([line[1], line[2]])
            x = np.array(x)
            scale = max(x[:, 0].max() - x[:, 0].min(), x[:, 1].max() -x[:, 1].min()) * (1 + 2 * 1e-4)
            x = x - x.min(0).reshape(1, 2)
            x = x / scale
            x = x + 1e-4
            if x[:, 0].max() > x[:, 1].max():
                x[:, 1] += (1 - 1e-4 - x[:, 1].max()) / 2
            else:
                x[:, 0] += (1 - 1e-4 - x[:, 0].max()) / 2
            x = x.reshape(1, n_nodes, 2)
        n_edges = 20
        batch_size = 1
        node_feat = x
        dist = x.reshape(batch_size, n_nodes, 1, 2) - x.reshape(batch_size, 1, n_nodes, 2)
        dist = np.sqrt((dist ** 2).sum(-1))

        edge_index = np.argsort(dist, -1)[:, :, 1:1 + n_edges]
        edge_feat = dist[np.arange(batch_size).reshape(-1, 1, 1), np.arange(n_nodes).reshape(1, -1, 1), edge_index]
        inverse_edge_index = -np.ones(shape=[batch_size, n_nodes, n_nodes], dtype="int")
        inverse_edge_index[np.arange(batch_size).reshape(-1, 1, 1), edge_index, np.arange(n_nodes).reshape(1, -1, 1)] = np.arange(n_edges).reshape(1, 1, -1) + np.arange(n_nodes).reshape(1, -1, 1) * n_edges
        inverse_edge_index = inverse_edge_index[np.arange(batch_size).reshape(-1, 1, 1), np.arange(n_nodes).reshape(1, -1, 1), edge_index]
        edge_index_np = edge_index

        node_feat = Variable(torch.FloatTensor(node_feat).type(torch.cuda.FloatTensor), requires_grad=False) # B x N x 2
        edge_feat = Variable(torch.FloatTensor(edge_feat).type(torch.cuda.FloatTensor), requires_grad=False).view(batch_size, -1, 1) # B x 20N x 1
        edge_index = Variable(torch.LongTensor(edge_index).type(torch.cuda.LongTensor), requires_grad=False).view(batch_size, -1) # B x 20N
        inverse_edge_index = Variable(torch.FloatTensor(inverse_edge_index).type(torch.cuda.LongTensor), requires_grad=False).view(batch_size, -1) # B x 20N
        candidate_test = []
        label = None
        edge_cw = None

        y_edges, _, y_nodes = net.forward(node_feat, edge_feat, edge_index, inverse_edge_index, label, edge_cw, n_edges)
        y_edges = y_edges.detach().cpu().numpy()
        y_edges = y_edges[:, :, 1].reshape(batch_size, n_nodes, n_edges)
        y_edges = np.argsort(-y_edges, -1)
        edge_index = np.array(edge_index_np)
        candidate_index = edge_index[np.arange(batch_size).reshape(-1, 1, 1), np.arange(n_nodes).reshape(1, -1, 1), y_edges]
        candidate_test.append(candidate_index[:, :, :5])
        candidate_test = np.concatenate(candidate_test, 0)
        with open("/home/appuser/NeuroLKH/result/tsplib/candidate/" + instance_name + ".txt", "w") as f:
            f.write(str(n_nodes) + "\n")
            for j in range(n_nodes):
                line = str(j + 1) + " 0 5"
                for _ in range(5):
                    line += " " + str(candidate_test[0, j, _] + 1) + " 1"
                f.write(line + "\n")
            f.write("-1\nEOF\n")

def solve_NeuroLKH(instance_name, opt_value, tsplib):
    start = time.time()
    para_filename = "/home/appuser/NeuroLKH/result/tsplib/NeuroLKH_para/" + instance_name + ".para"
    log_filename = "/home/appuser/NeuroLKH/result/tsplib/NeuroLKH_log/" + instance_name + ".log"
    write_para(instance_name, "NeuroLKH", para_filename, opt_value, tsplib)
    with open(log_filename, "w") as f:
        check_call(["/home/appuser/NeuroLKH/LKH", para_filename], stdout=f)
    end = time.time()
    execution = end-start
    print("\nLKH:", execution)

    return read_results(log_filename)

def read_results(log_filename):
    results = []
    with open(log_filename, "r") as f:
        lines = f.readlines()
        print(lines[-7])
        successes = int(lines[-7].split(" ")[-2].split("/")[0])
        cost_min = float(lines[-6].split(",")[0].split(" ")[-1])
        cost_avg = float(lines[-6].split(",")[1].split(" ")[-1])
        trials_min = float(lines[-4].split(",")[0].split(" ")[-1])
        trials_avg = float(lines[-4].split(",")[1].split(" ")[-1])
        time = float(lines[-3].split(",")[1].split(" ")[-2])
        return successes, cost_min, cost_avg, trials_min, trials_avg, time

def eval_dataset(instance_names, method, args, opt_values):
    os.makedirs("/home/appuser/NeuroLKH/result/tsplib/" + method + "_para", exist_ok=True)
    os.makedirs("/home/appuser/NeuroLKH/result/tsplib/" + method + "_log", exist_ok=True)
    if method == "NeuroLKH":
        os.makedirs("/home/appuser/NeuroLKH/result/tsplib/candidate", exist_ok=True)
        net = SparseGCNModel()
        net.cuda()
        saved = torch.load(args.model_path)
        net.load_state_dict(saved["model"])
        sgn_start_time = time.time()
        with torch.no_grad():
            start = time.time()

            print(args.tsplib)

            if args.tsplib == True:
                infer_SGN_write_candidate(net, instance_names)
            else:
                infer_SGN_write_candidate2(net, instance_names)
            end = time.time()
            execution = end-start
            print("Inference:", execution)
        sgn_runtime = time.time() - sgn_start_time
        with Pool(os.cpu_count()) as pool:
            results = list(tqdm.tqdm(pool.imap(method_wrapper, [("NeuroLKH", instance_names[i], opt_values[instance_names[i]], args.tsplib) for i in range(len(instance_names))]), total=len(instance_names)))
    else:
        assert method == "LKH"
        feat_runtime = 0
        sgn_runtime = 0
        with Pool(os.cpu_count()) as pool:
            results = list(tqdm.tqdm(pool.imap(method_wrapper, [("LKH", instance_names[i], opt_values[instance_names[i]]) for i in range(len(instance_names))]), total=len(instance_names)))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_path', type=str, default='/home/appuser/NeuroLKH/pretrained/neurolkh.pt', help='')
    parser.add_argument('--n_samples', type=int, default=1, help='')
    parser.add_argument('--tsplib', action='store_true', help='')
    parser.add_argument('--instance_name', type=str, default='eil76', help='')
    parser.add_argument('--optimal_value', type=int, default='10000', help='')
    args = parser.parse_args()
    instance_names = []
    instance_names.append(args.instance_name)

    with open("/home/appuser/NeuroLKH/tsplib_data/opt.pkl", "rb") as f:
        opt_values = pickle.load(f)
    opt_values[instance_names[0]] = args.optimal_value
    
    args.model_path = "/home/appuser/NeuroLKH/pretrained/neurolkh_m.pt"

    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    print("Current Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    start = time.time()

    neurolkh_m_results = eval_dataset(instance_names, "NeuroLKH", args, opt_values)

    print ("Successes Best Avgerage Trials_Min Trials_Avg Time")

    for i in range(len(neurolkh_m_results)):
        print ("------%s------" % (instance_names[i]))
        print (neurolkh_m_results[i])

    end = time.time()
    execution = end-start
    print("Overall:", execution)
