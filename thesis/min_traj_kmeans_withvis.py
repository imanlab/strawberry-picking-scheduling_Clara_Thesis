import os
import networkx as nx
import numpy as np
import itertools
from more_itertools import pairwise
import csv
import math
import pandas as pd
import cv2
from sklearn.cluster import KMeans

def get_coordinates(file_path):
    coordinates = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for index, row in enumerate(reader, start=0):
            x = round(float(row[0]))
            y = round(float(row[1]))
            z = round(float(row[2])) if len(row) > 2 and row[2] else None  # Set z to None if it's missing
            coordinates.append((index, (x, y, z)))
    return coordinates

def find_centroids(coordinates, n_clusters=4):
    """Find centroids using KMeans clustering."""
    coords = np.array([coord[1] for coord in coordinates if coord[1][2] is not None])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords[:, :2])
    centroids = kmeans.cluster_centers_
    return centroids

def compute_optimal_path(file_path):
    all_coordinates = get_coordinates(file_path)
    
    # Initialize the graph
    G = nx.Graph()
    
    # Add all points as nodes to the graph using the original index as the node label
    for index, coord in all_coordinates:
        if coord[2] is not None:  # Only add nodes with a valid z-coordinate
            G.add_node(index, pos=coord)

    # Add edges to the graph
    for u, v in itertools.combinations(G.nodes, 2):
        if u != v:
            G.add_edge(u, v, weight=np.linalg.norm(np.array(G.nodes[u]['pos']) - np.array(G.nodes[v]['pos'])))

    # Filter out nodes with a z-coordinate of zero
    valid_coordinates = [coord for coord in all_coordinates if coord[1][2] is not None]

    # Calculate the minimum and maximum coordinates for each dimension 
    min_coords = [min(coord[1][axis] for coord in valid_coordinates) for axis in range(3)]
    max_coords = [max(coord[1][axis] for coord in valid_coordinates) for axis in range(3)]
    
    # Calculate the total length in each dimension
    total_length_x = max_coords[0] - min_coords[0]
    total_length_y = max_coords[1] - min_coords[1]
    total_length_z = max_coords[2] - min_coords[2]

    # Determine the segment length for each dimension
    mid_length_x = (total_length_x / 2) + min_coords[0]
    mid_length_y = (total_length_y / 2) + min_coords[1]

    # Initialize the segments
    segments = [[], [], [], []]

    # Function to add a node to a segment based on the given length
    def add_to_segment(segment, mid_length_x, mid_length_y, node):
        x, y, z = G.nodes[node]['pos']

        # Check which segment the node belongs to based on its coordinates
        if x < mid_length_x and y < mid_length_y:
            segment[0].append(node)
        elif x >= mid_length_x and y < mid_length_y:
            segment[1].append(node)
        elif x < mid_length_x and y >= mid_length_y:
            segment[2].append(node)
        elif x >= mid_length_x and y >= mid_length_y:
            segment[3].append(node)

    # Iterate through the nodes and add them to segments based on their coordinates
    for node in G.nodes():
        add_to_segment(segments, mid_length_x, mid_length_y, node)

    # Create subgraphs
    G1 = G.subgraph(set(segments[0]))
    G2 = G.subgraph(set(segments[1]))
    G3 = G.subgraph(set(segments[2]))
    G4 = G.subgraph(set(segments[3]))

    # Generate all permutations of the nodes for each part
    permutations1 = list(itertools.permutations(G1.nodes()))
    permutations2 = list(itertools.permutations(G2.nodes()))
    permutations3 = list(itertools.permutations(G3.nodes()))
    permutations4 = list(itertools.permutations(G4.nodes()))

    graphs = [permutations1, permutations2, permutations3, permutations4]
    permutations_graphs = list(itertools.permutations(graphs))

    # Calculate the total distance for each permutation for each part
    distances1 = [(perm, sum(G1.edges[u, v]['weight'] for u, v in pairwise(perm))) for perm in permutations1]
    distances2 = [(perm, sum(G2.edges[u, v]['weight'] for u, v in pairwise(perm))) for perm in permutations2]
    distances3 = [(perm, sum(G3.edges[u, v]['weight'] for u, v in pairwise(perm))) for perm in permutations3]
    distances4 = [(perm, sum(G4.edges[u, v]['weight'] for u, v in pairwise(perm))) for perm in permutations4]

    # Sort the permutations by their total distance for each part
    sorted_distances1 = sorted(distances1, key=lambda x: x[1])[:1]
    sorted_distances2 = sorted(distances2, key=lambda x: x[1])[:1]
    sorted_distances3 = sorted(distances3, key=lambda x: x[1])[:1]
    sorted_distances4 = sorted(distances4, key=lambda x: x[1])[:1]

    sorted_distances = [sorted_distances1, sorted_distances2, sorted_distances3, sorted_distances4]
    miao = list(itertools.permutations(sorted_distances))

    # Initialize an empty list to store the first elements of each combination
    permutations_distances = []
    first_elements = []

    # Iterate over each combination in the main list
    for combination in miao:
        # Initialize an empty list to store the first elements of the current combination
        first_elements_of_combination = []

        # Iterate over each tuple within the current combination
        for group in combination:
            # Skip empty groups
            if group:
                # Extract the first element of the each combination
                first_element = group[0]
                first_elements_of_combination.append((first_element))

        # Add the first elements of the current combination to the main list
        permutations_distances.append(first_elements_of_combination)

    # Calculate total distances for each combination
    tuplas_totalone = []
    for combination in permutations_distances:
        total_distance = 0
        total_distance_among_subgraph = 0
        total_distance_in_the_subgraph1 = 0
        total_distance_in_the_subgraph2 = 0
        path = []
        tuplas = []
        last_non_none_node = None
        connection_weight = 0

        for i in range(len(combination) - 1):
            # Get the last node of the current tuple
            last_node_current = combination[i][0][-1] if combination[i][0] else None

            # If last_node_current is None, use the last_non_none_node for the connection_weight computation
            if last_node_current is None:
                last_node_current = last_non_none_node
            else:
                last_non_none_node = last_node_current  # Update the last non-None node

            total_distance_in_the_subgraph1 += combination[i][1]
            path.extend(combination[i][0])

            # Get the first node of the next tuple
            first_node_next = combination[i + 1][0][0] if combination[i + 1][0] else None

            # If first_node_next is None, use the last_non_none_node for the connection_weight computation
            if first_node_next is None:
                first_node_next = last_non_none_node
                continue
            else:
                last_non_none_node = first_node_next  # Update the last non-None node

            # Calculate the weight of the connection between the two subgraphs
            if last_node_current in G.nodes and first_node_next in G.nodes:
                connection_weight = G.edges[last_node_current, first_node_next]['weight']
                total_distance_among_subgraph += connection_weight

            total_distance_in_the_subgraph2 = combination[i + 1][1]
            total_distance = total_distance_in_the_subgraph1 + total_distance_in_the_subgraph2 + total_distance_among_subgraph

        # Add the path of the last tuple to the combined path
        path.extend(combination[-1][0])

        # Append the combined path and total distance to the tuplas list
        tuplas.append([path, total_distance])
        tuplas_totalone.append(tuplas)

    # Sort and get the top 10 combinations
    sorted_tuple = sorted(tuplas_totalone, key=lambda x: x[0][1])
    top_ten_elements = sorted_tuple[:10]

    # Define top_ten_tuples based on sorted_tuple
    top_ten_tuples = [t[0] for t in sorted_tuple[:10]]
    
    # Find centroids
    centroids = find_centroids(all_coordinates)
    
    # Return the optimal path (the first path in the sorted top ten elements), the graph, the segments, and the centroids
    return top_ten_tuples[0][0], G, segments, centroids

def process_single_csv(file_path, image_path, output_path):
    # Compute the optimal path and get the graph
    optimal_path, G, segments, centroids = compute_optimal_path(file_path)
    
    # Visualization of the path on the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Define colors for each segment
    segment_colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)]

    # Draw the clusters on the image
    for i, segment in enumerate(segments):
        color = segment_colors[i]
        for node in segment:
            coord = G.nodes[node]['pos']
            point = (int(coord[0]), int(coord[1]))
            cv2.circle(image, point, 5, color, -1)

    # Draw the centroids on the image
    for centroid in centroids:
        point = (int(centroid[0]), int(centroid[1]))
        cv2.circle(image, point, 200, (0, 255, 255), 2)  # Draw a larger circle for centroids

    # Draw the path on the image and add text labels
    for i in range(len(optimal_path)):
        current_node = optimal_path[i]
        coord = G.nodes[current_node]['pos']
        point = (int(coord[0]), int(coord[1]))
        
        # Draw the point
        cv2.circle(image, point, 5, (0, 0, 255), 1)
        
        # Add the index label of the strawberry
        cv2.putText(image, f'{current_node}', (point[0] + 10, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Add the order number in the path
        cv2.putText(image, f'{i+1}', (point[0] + 10, point[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if i > 0:
            prev_node = optimal_path[i - 1]
            prev_point = (int(G.nodes[prev_node]['pos'][0]), int(G.nodes[prev_node]['pos'][1]))
            image = cv2.line(image, prev_point, point, (0, 255, 0), 2)

    # Save the image with the path drawn on it
    output_image_path = os.path.join(output_path, 'output_image_with_path.png')
    cv2.imwrite(output_image_path, image)
    
    # Show the image with the path
    cv2.imshow('Path Visualization', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
csv_file_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/coordinates_for_images/73.csv'  # Update with your actual CSV file path
image_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/strawberry_riseholme_lincoln_tbd_73_rgb.png'  # Update with your actual image file path
output_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/visualization'  # Directory where you want to save the output

process_single_csv(csv_file_path, image_path, output_path)

