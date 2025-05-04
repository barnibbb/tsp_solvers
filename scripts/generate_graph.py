import numpy as np
import cv2
import random
import sys
import os


def generate_graph(num_nodes=50, x_range=(0,1000), y_range=(0,1000), seed=None):

    if seed is not None:
        np.random.seed(seed)

    positions = []

    while len(positions) < num_nodes:
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        point = np.array([x,y])
        
        if all(np.linalg.norm(point - np.array(p)) >= 20 for p in positions):
            positions.append(point)

    return np.array(positions).astype(int)


def draw_graph(positions, img_size=(1000, 1000)):
    img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255

    num_nodes = positions.shape[0]

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            pt1 = tuple(positions[i])
            pt2 = tuple(positions[j])
            cv2.line(img, pt1, pt2, (200, 200, 200), 1)

    for i, (x, y) in enumerate(positions):
        cv2.circle(img, (x,y), 5, (0, 0, 255), -1)
        cv2.putText(img, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

    return img


def export_to_tsplib(positions, filename):

    name = os.path.splitext(os.path.basename(filename))[0]

    with open(filename, 'w') as f:
        f.write(f"NAME: {name}\n")
        f.write("TYPE: TSP\n")
        f.write("COMMENT: Random generated graph\n")
        f.write(f"DIMENSION: {num_nodes}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(positions):
            f.write(f"{i + 1} {x} {y}\n")
        f.write("EOF\n")


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python3 generate_graph.py <output_folder> <num_nodes>")
        sys.exit(1)

    output_folder = sys.argv[1]

    os.makedirs(output_folder, exist_ok=True)

    num_nodes = int(sys.argv[2])

    output_file = os.path.join(output_folder, "generated_graph.tsp")

    positions = generate_graph(num_nodes=num_nodes, x_range=(50,950), y_range=(50,950))
    img = draw_graph(positions)

    cv2.imshow("Complete graph", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    export_to_tsplib(positions, output_file)
