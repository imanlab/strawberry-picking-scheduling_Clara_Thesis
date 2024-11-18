import os
import numpy as np
import csv
import itertools
import networkx as nx
import cv2

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

def visualize_two_paths_with_legend(file_path, image_path, output_path, path1, path2):
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

    # Visualization of the paths on the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Define colors for the paths
    path_colors = [(0, 255, 0), (255, 0, 0)]  # Red for path1 (easier), Blue for path2 (shorter)

    # Draw the first path (path1 - easier)
    for i in range(len(path1)):
        current_node = path1[i]
        coord = G.nodes[current_node]['pos']
        point = (int(coord[0]), int(coord[1]))
        
        # Draw the point
        cv2.circle(image, point, 5, (0, 0, 0), -1)
        
        # Add the order number for path1
        cv2.putText(image, f'{i+1}', (point[0] + 10, point[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, path_colors[0], 2)
        
        if i > 0:
            prev_node = path1[i - 1]
            prev_point = (int(G.nodes[prev_node]['pos'][0]), int(G.nodes[prev_node]['pos'][1]))
            image = cv2.line(image, prev_point, point, path_colors[0], 2)

    # Draw the second path (path2 - shorter)
    for i in range(len(path2)):
        current_node = path2[i]
        coord = G.nodes[current_node]['pos']
        point = (int(coord[0]), int(coord[1]))
        
        # Draw the point
        cv2.circle(image, point, 5, (0, 0, 0), -1)
        
        # Add the order number for path2
        cv2.putText(image, f'{i+1}', (point[0] - 10, point[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, path_colors[1], 2)
        
        if i > 0:
            prev_node = path2[i - 1]
            prev_point = (int(G.nodes[prev_node]['pos'][0]), int(G.nodes[prev_node]['pos'][1]))
            image = cv2.line(image, prev_point, point, path_colors[1], 2)

    # Add a legend to the image
    legend_start_x = 50
    legend_start_y = 50
    legend_spacing = 40
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Path 1 legend (easier)
    cv2.rectangle(image, (legend_start_x, legend_start_y), (legend_start_x + 30, legend_start_y + 30), path_colors[0], -1)
    cv2.putText(image, 'Path 1: Easier', (legend_start_x + 40, legend_start_y + 25), font, 1, (0, 255, 0), 2)

    # Path 2 legend (shorter)
    cv2.rectangle(image, (legend_start_x, legend_start_y + legend_spacing), (legend_start_x + 30, legend_start_y + legend_spacing + 30), path_colors[1], -1)
    cv2.putText(image, 'Path 2: Shorter', (legend_start_x + 40, legend_start_y + legend_spacing + 25), font, 1, (255, 0, 0), 2)

    # Save the image with both paths drawn on it and the legend
    output_image_path = os.path.join(output_path, 'output_image_10.png')
    cv2.imwrite(output_image_path, image)
    
    # Show the image with both paths and the legend
    cv2.imshow('Path Visualization', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
csv_file_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/coordinates_for_images/10.csv'  # Update with your actual CSV file path
image_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/strawberry_riseholme_lincoln_tbd_10_rgb.png'  # Update with your actual image file path
output_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/visualization'  # Directory where you want to save the output

# Example given paths
path1 = [7, 8, 2, 5, 4, 6, 1, 0

]  # Update with your actual first path (easier)
path2 = [8, 7, 5, 2, 6, 4, 1, 0
] # Update with your actual second path (shorter)

visualize_two_paths_with_legend(csv_file_path, image_path, output_path, path1, path2)

