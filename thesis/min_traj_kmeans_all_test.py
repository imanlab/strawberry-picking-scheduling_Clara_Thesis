import networkx as nx
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from more_itertools import pairwise
from collections import defaultdict
import csv
import itertools
from sklearn.neighbors import KNeighborsClassifier
import math
import os
import time  # Import time module


#Optimized trajectory kmeans + brut force

def get_coordinates(file_path):
    coordinates = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        #next(reader)  # Skip header row if present
        for index, row in enumerate(reader, start=0):
            # Append the index and the coordinate tuple to the list, even if the z-coordinate is missing
            x = round(float(row[0]))
            y = round(float(row[1]))
            z = round(float(row[2])) if row[2] else None  # Set z to None if it's missing
            coordinates.append((index, (x, y, z)))
    return coordinates


def find_centroids(coordinates, n_clusters=4):

    # Ensure the data is in the correct format for K-means
    X = np.array(coordinates, dtype=object)
    #print(X)

    # Instantiate KMeans with the desired number of clusters
    kmeans = KMeans(n_clusters=n_clusters)

    # Fit KMeans to the data
    kmeans.fit(X)

    # Get the cluster labels for each data point
    labels = kmeans.labels_
    
    # Get the coordinates of the cluster centroids
    centroids = kmeans.cluster_centers_
    #print(centroids)
    #print(f"labels: {labels}")

    return labels, centroids
    
def euclidean_distance(point1, point2):
    # Assuming point1 and point2 are tuples of coordinates (x, y, z)
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def assign_nodes_to_clusters(nodes, centroids):
    clusters = {i: [] for i in range(len(centroids))}
    cluster_sizes = {i:  0 for i in range(len(centroids))}
    
    for node_id, node_coords in nodes:
        min_distance = float('inf')
        min_centroid_index = -1
        second_min_distance = float('inf')
        second_min_centroid_index = -1
        
        for centroid_index, centroid_coords in enumerate(centroids):
            distance = euclidean_distance(node_coords, centroid_coords)
            if distance < min_distance:
                second_min_distance = min_distance
                second_min_centroid_index = min_centroid_index
                min_distance = distance
                min_centroid_index = centroid_index
            elif distance < second_min_distance:
                second_min_distance = distance
                second_min_centroid_index = centroid_index
        
        # If the cluster with the min distance has less than  10 nodes, add the node to that cluster
        if cluster_sizes[min_centroid_index] <  10:
            clusters[min_centroid_index].append(node_id)
            cluster_sizes[min_centroid_index] +=  1
        # If the cluster with the min distance has  10 nodes, try to add the node to the cluster with the second min distance
        elif cluster_sizes[second_min_centroid_index] <  10:
            clusters[second_min_centroid_index].append(node_id)
            cluster_sizes[second_min_centroid_index] +=  1
        # If all clusters have  10 nodes, you may choose to add the node to the cluster with the smallest distance or leave it unassigned
        # For demonstration, let's add it to the cluster with the smallest distance
        else:
            clusters[min_centroid_index].append(node_id)
            cluster_sizes[min_centroid_index] +=  1
    
    return clusters

# Define input and output directories
input_directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/test/coordinates_for_images'
output_directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/test/optimal_paths_kmeans'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

start_time = time.time()  # Start timing

# Iterate over each file in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_directory, filename)
        all_coordinates = get_coordinates(file_path)
        
        # Initialize the graph
        G = nx.Graph()

        # Add all points as nodes to the graph using the original index as the node label
        for index, coord in all_coordinates:
            #print(index)
            #print(coord)
            if coord[2] is not None:  # Only add nodes with a valid z-coordinate
                G.add_node(index, pos=coord)
                #print(index)
                #print(coord)

        print(G.nodes)
        
        if len(G.nodes) <  4:
          # Since the graph has fewer than  4 nodes, we can directly compute the distance for the entire graph
          # Add edges to the graph if not already present
          for u, v in itertools.combinations(G.nodes,  2):
              if u != v:
                   G.add_edge(u, v, weight=np.linalg.norm(np.array(G.nodes[u]['pos']) - np.array(G.nodes[v]['pos'])))
    
          # Generate all permutations of the nodes
          permutations_single = list(itertools.permutations(G.nodes))
    
          # Calculate the total distance for each permutation
          total_distance_single_permutation = [(perm, sum(G.edges[u, v]['weight'] for u, v in pairwise(perm))) for perm in permutations_single]
        
          # Sort the permutations by their total distance and take the shortest one
          shortest_path = sorted(total_distance_single_permutation, key=lambda x: x[1])[:10]
          print(shortest_path[0])
    
          # Prepare the output: path and total distance
          path_str = ', '.join(map(str, shortest_path[0]))
          #total_distances_sorted = [t[0] for t in shortest_path[:10]]
          
    
          # Define output filename
          base_filename = os.path.basename(file_path)
          filename_without_extension, _ = os.path.splitext(base_filename)
          output_filename = f"{filename_without_extension}_results.csv"
          output_file_path = os.path.join(output_directory, output_filename)
    
          # Write the shortest path and total distance to the output file
          with open(output_file_path, 'w', newline='') as csvfile:
              csvwriter = csv.writer(csvfile)
              csvwriter.writerow(['Path', 'Total Distance'])
              #csvwriter.writerow([path_str, shortest_path])
              for i, (path_list, distance) in enumerate(shortest_path, start=1):
                     path_str = ', '.join(map(str, path_list))
                     csvwriter.writerow([path_str, distance]) 
                            
        
        else:
        
            # Add edges to the graph
            for u, v in itertools.combinations(G.nodes,  2):
                if u != v:
                    G.add_edge(u, v, weight=np.linalg.norm(np.array(G.nodes[u]['pos']) - np.array(G.nodes[v]['pos'])))
            #print(G)

            # Filter out nodes with a z-coordinate of zero
            valid_coordinates = [coord for coord in all_coordinates if coord[1][2] is not None]
            #print(f"le valid coordinates: {valid_coordinates}")
            num_subgraphs =  4

            X = np.array([coord[1] for coord in valid_coordinates], dtype=float)
            #print(f" X: {X}")

            #nodes = np.array([coord[0] for coord in valid_coordinates], dtype=float)

            # Calculate the number of nodes for each cluster size
            n_nodes = len(X)
            #print(f"n_nodes: {n_nodes}")

            n_clusters =  num_subgraphs 
  
            labels, centroids = find_centroids(X, n_clusters)
            clusters = assign_nodes_to_clusters(valid_coordinates, centroids)

            #print(clusters)

            # Create a dictionary to map indices to subgraphs
            # Initialize the subgraphs
       
            # Replace the old subgraph creation with the new segments
            G1 = G.subgraph(clusters[0])
            G2 = G.subgraph(clusters[1])
            G3 = G.subgraph(clusters[2])
            G4 = G.subgraph(clusters[3])    

            #print(G1)
            #print(G2)
            #print(G3)
            #print(G4)   
        
            # Generate all permutations of the nodes for each part
            permutations1 = list(itertools.permutations(G1.nodes()))
            permutations2 = list(itertools.permutations(G2.nodes()))
            permutations3 = list(itertools.permutations(G3.nodes()))
            permutations4 = list(itertools.permutations(G4.nodes()))
            #print(permutations1)
            #print(permutations2)
            #print(permutations3)
            #print(permutations4)

            graphs = [permutations1, permutations2, permutations3, permutations4]
  
            permutations_graphs = list(itertools.permutations(graphs))

            #print(f'print the graph permutations: {permutations_graphs}')

            #permutations_graphs = list(itertools.permutations(permutations1,  permutations2, permutations3, permutations4))

            #print(permutations1)
            #print(permutations2)
            #print(permutations3)
            #print(permutations4)

            # Calculate the total distance for each permutation for each part
            distances1 = [(perm, sum(G1.edges[u, v]['weight'] for u, v in pairwise(perm))) for perm in permutations1]
            distances2 = [(perm, sum(G2.edges[u, v]['weight'] for u, v in pairwise(perm))) for perm in permutations2]
            distances3 = [(perm, sum(G3.edges[u, v]['weight'] for u, v in pairwise(perm))) for perm in permutations3]
            distances4 = [(perm, sum(G4.edges[u, v]['weight'] for u, v in pairwise(perm))) for perm in permutations4]

            #print(distances1)
            #print(distances2)
            #print(distances3)
            #print(distances4)

            # Sort the permutations by their total distance for each part
            sorted_distances1 = sorted(distances1, key=lambda x: x[1])[:1]
            sorted_distances2 = sorted(distances2, key=lambda x: x[1])[:1]
            sorted_distances3 = sorted(distances3, key=lambda x: x[1])[:1]
            sorted_distances4 = sorted(distances4, key=lambda x: x[1])[:1]
 
            sorted_distances = [sorted_distances1, sorted_distances2, sorted_distances3, sorted_distances4]
            #print(f"le sorted_distances: {sorted_distances}")
            #print(f'print the graph permutations: {permutations_graphs}')
            # Generate all possible combinations of the top  10 solutions from each partition

            miao = list(itertools.permutations(sorted_distances))
            #print(f"printiamo miao: {miao}")

            # Initialize an empty list to store the first elements of each combination
            permutations_distances =[]
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

            # Now, first_elements will contain the first elements of each combination
            #print(f"print combinations giuste?: {permutations_distances}")

            # Calculate total distances for each combination
            #Assumiamo di avere tutte le combinazioni che vogliamo per ora
            tuplas_totalone = []   
            for combination in permutations_distances:
               total_distance =  0
               
               total_distance_among_subgraph = 0
               total_distance_in_the_subgraph1 = 0
               total_distance_in_the_subgraph2 = 0
               path = []
               tuplas = []
               last_non_none_node = None
               connection_weight = 0

               for i in range(len(combination) -  1):
                    # Get the last node of the current tuple
                    last_node_current = combination[i][0][-1] if combination[i][0] else None
        
                    # If last_node_current is None, use the last_non_none_node for the connection_weight computation
                
                    if last_node_current is None:
                        last_node_current = last_non_none_node
            
                    else:
                        last_non_none_node = last_node_current  # Update the last non-None node
                        #print(f"last non node: {last_non_none_node}")
            
                    #print(f"print last_node_current: {last_node_current}")
                    total_distance_in_the_subgraph1 += combination[i][1]
                    #print(f"total_distance_in_the_subgraph1: {total_distance_in_the_subgraph1}") 
                    path.extend(combination[i][0])     
                    #print(path)
          
                    # Get the first node of the next tuple
                    first_node_next = combination[i +  1][0][0] if combination[i +  1][0] else None
                    #print(i)
                    # Skip to the next iteration if either of the nodes is None
                
                    # If first_node_next is None, use the last_non_none_node for the connection_weight computation
                    if first_node_next is None:
                         first_node_next = last_non_none_node
                         continue
                    else:
                         last_non_none_node = first_node_next  # Update the last non-None node
                         #print(f"last non node: {last_non_none_node}")

                    #print(f"print first_node_next: {first_node_next}")
        
                    # Calculate the weight of the connection between the two subgraphs
                    # only if both nodes are not None
                    if last_node_current in G.nodes and first_node_next in G.nodes:
                          connection_weight = G.edges[last_node_current, first_node_next]['weight']
                          total_distance_among_subgraph += connection_weight
            
                    #print(f"connection weight: {connection_weight}")
                    #print(f"somma connection weight: {total_distance_among_subgraph}")
                    total_distance_in_the_subgraph2 = combination[i+1][1]
        
                    #print(f"total_distance_in_the_subgraph2: {total_distance_in_the_subgraph2}")
                    total_distance = total_distance_in_the_subgraph1 + total_distance_in_the_subgraph2 + total_distance_among_subgraph
                    #print(f"distanza per ciclo: {total_distance}") 

               # Add the path of the last tuple to the combined path
               path.extend(combination[-1][0])
               #print(path)
    
               # Append the combined path and total distance to the tuplas list
               tuplas.append([path, total_distance])
               #print(tuplas)
               tuplas_totalone.append(tuplas)
            #print(f"le totalone {tuplas_totalone}")
    
            #esci dal for e sorted e prendi i primi 10 e stampa
            sorted_tuple = sorted(tuplas_totalone, key=lambda x: x[0][1])
            #print(f"le sortie: {sorted_tuple}")

            # Extract the first  10 elements from sorted_tuple
            top_ten_elements = sorted_tuple[:10]
            #print(f"i top {top_ten_elements}")

            # Define top_ten_tuples based on sorted_tuple
            top_ten_tuples = [t[0] for t in sorted_tuple[:10]]

            # Use the input filename to create a unique output filename
            base_filename = os.path.basename(file_path)
            filename_without_extension, _ = os.path.splitext(base_filename)
            output_filename = f"{filename_without_extension}_results.csv"
            output_file_path = os.path.join(output_directory, output_filename)
            with open(output_file_path, 'w', newline='') as csvfile:
                  csvwriter = csv.writer(csvfile)
                  csvwriter.writerow(['Path', 'Total Distance'])
                  for i, (path_list, distance) in enumerate(top_ten_tuples, start=1):
                     path_str = ', '.join(map(str, path_list))
                     csvwriter.writerow([path_str, distance])
        
end_time = time.time()  # End timing
total_time = end_time - start_time
print(f'process complete in: {total_time:.2f} seconds')       

