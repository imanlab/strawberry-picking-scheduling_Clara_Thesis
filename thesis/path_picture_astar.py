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

def visualize_path(file_path, image_path, output_path, given_path):
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

    # Create segments based on coordinates
    segments = [[], [], [], []]
    mid_length_x = (max(coord[1][0] for coord in all_coordinates) - min(coord[1][0] for coord in all_coordinates)) / 2
    mid_length_y = (max(coord[1][1] for coord in all_coordinates) - min(coord[1][1] for coord in all_coordinates)) / 2

    def add_to_segment(segment, mid_length_x, mid_length_y, node):
        x, y, z = G.nodes[node]['pos']
        if x < mid_length_x and y < mid_length_y:
            segment[0].append(node)
        elif x >= mid_length_x and y < mid_length_y:
            segment[1].append(node)
        elif x < mid_length_x and y >= mid_length_y:
            segment[2].append(node)
        elif x >= mid_length_x and y >= mid_length_y:
            segment[3].append(node)

    for node in G.nodes():
        add_to_segment(segments, mid_length_x, mid_length_y, node)
    
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

    # Draw the given path on the image and add text labels
    for i in range(len(given_path)):
        current_node = given_path[i]
        coord = G.nodes[current_node]['pos']
        point = (int(coord[0]), int(coord[1]))
        
        # Draw the point
        cv2.circle(image, point, 5, (0, 0, 0), 1)
        
        '''
        # Add the index label of the strawberry
        cv2.putText(image, f'{current_node}', (point[0] + 10, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        '''
        # Add the order number in the path
        cv2.putText(image, f'{i+1}', (point[0] + 10, point[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        if i > 0:
            prev_node = given_path[i - 1]
            prev_point = (int(G.nodes[prev_node]['pos'][0]), int(G.nodes[prev_node]['pos'][1]))
            image = cv2.line(image, prev_point, point, (0, 0, 0), 2)

    # Save the image with the path drawn on it
    output_image_path = os.path.join(output_path, 'output_image_with_path_10_shortest.png')
    cv2.imwrite(output_image_path, image)
    
    # Show the image with the path
    cv2.imshow('Path Visualization', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
csv_file_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/coordinates_for_images/10.csv'  # Update with your actual CSV file path
image_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/strawberry_riseholme_lincoln_tbd_10_rgb.png'  # Update with your actual image file path
output_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/visualization'  # Directory where you want to save the output

# Example given path
given_path = [8, 7, 5, 2, 6, 4, 1, 0





]  # Update with your actual given path

visualize_path(csv_file_path, image_path, output_path, given_path)

