import numpy as np
import pandas as pd
import networkx as nx
import heapq
from collections import defaultdict
import glob
import os
import csv
import itertools
import random
import time  # Import time module


def get_coordinates(file_path):
    coordinates = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for index, row in enumerate(reader, start=0):
            x = round(float(row[0]))
            y = round(float(row[1]))
            z = round(float(row[2])) if row[2] else None
            coordinates.append((index, (x, y, z)))
    return coordinates

def distance(coord1, coord2):
    return np.sqrt(np.sum((np.array(coord1) - np.array(coord2))**2))

def heuristic(state, remaining_coords):
    return sum(distance(state[0], coord) for coord in remaining_coords)

def a_star_tsp(coords, initial_node):
    num_nodes = len(coords)
    initial_state = (coords[initial_node][1], 0, tuple(node for node in range(num_nodes) if node != initial_node))
    queue = [(heuristic(initial_state, initial_state[2]), initial_state)]
    visited = set()
    node_visited = set()
    predecessors = {}
    total_distance = 0

    while queue:
        _, (current_node, total_distance, remaining) = heapq.heappop(queue)
        node_visited.add(current_node)

        if (current_node, remaining) in visited:
            continue
        visited.add((current_node, remaining))

        if current_node == coords[initial_node][1] and len(remaining) == 0:
            break

        next_node_candidates = [(distance(current_node, coords[next_node][1]), next_node) for next_node in remaining]
        next_node_candidates.sort()
        for _, next_node in next_node_candidates:
            if coords[next_node][1] not in node_visited:
                chosen_node = coords[next_node][1]
                break

        predecessors[next_node] = chosen_node

        new_remaining = tuple(node for node in remaining if node != next_node)
        new_cost = distance(current_node, chosen_node)
        total_distance += new_cost 
        h = heuristic((chosen_node, total_distance, new_remaining), new_remaining)
        total_estimated_cost = total_distance + h 

        heapq.heappush(queue, (total_estimated_cost, (chosen_node, total_distance, new_remaining)))

    path = [coords[initial_node][0]] + [coords[i][0] for i in predecessors.keys()]
    current_node = chosen_node

    return total_distance, node_visited, path

# Input folder containing CSV files
input_folder = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/coordinates_for_images'

# Output folder to save results
output_folder = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/optimal_path_astar'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get list of input CSV files
input_files = glob.glob(os.path.join(input_folder, '*.csv'))
start_time = time.time()  # Start timing
for file_path in input_files:
    # Get coordinates from input CSV
    coords = get_coordinates(file_path)
    valid_coordinates = [coord for coord in coords if coord[1][2] is not None]
    results = []

    # Handle case where there is only one valid coordinate
    if len(valid_coordinates) <= 1:
        node = valid_coordinates[0][0]
        # Print node and 0 as path
        print(f"File: {file_path}, Node: {node}, Path: {node}, Total Distance: 0")
        continue

    for _ in range(30):
        initial_node = random.randint(0, len(valid_coordinates) - 1)
        total_cost, node_visited, path = a_star_tsp(valid_coordinates, initial_node)
        results.append((path, total_cost))

    # Sort results by increasing total distance
    results.sort(key=lambda x: x[1])

    # Save the best 10 results
    top_10_results = results[:10]

    # Extract file name from path
    file_name = os.path.basename(file_path)
    # Generate output file path
    output_file_path = os.path.join(output_folder, f'results_{file_name}')

    # Write the best 10 results to CSV
    with open(output_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Path', 'Total Distance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for path, total_distance in top_10_results:
            writer.writerow({'Path': path, 'Total Distance': total_distance})

print("Processing complete.")
end_time = time.time()  # End timing
total_time = end_time - start_time
print(f'process complete in: {total_time:.2f} seconds') 
