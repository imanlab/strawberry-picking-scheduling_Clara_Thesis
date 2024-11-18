import os
import networkx as nx
import numpy as np
import itertools
from more_itertools import pairwise
from itertools import permutations
import csv
import matplotlib.pyplot as plt
import math

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

all_coordinates = get_coordinates('/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/coordinates_for_images/97.csv')
#print(all_coordinates)

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

#print(G)

# Add edges to the graph
for u, v in itertools.combinations(G.nodes,  2):
    if u != v:
        G.add_edge(u, v, weight=np.linalg.norm(np.array(G.nodes[u]['pos']) - np.array(G.nodes[v]['pos'])))
#print(G)
# Filter out nodes with a z-coordinate of zero
valid_coordinates = [coord for coord in all_coordinates if coord[1][2] is not None]
#print(valid_coordinates)
# Calculate the minimum and maximum coordinates for each dimension
min_coords = [min(coord[1][axis] for coord in valid_coordinates) for axis in range(3)]
max_coords = [max(coord[1][axis] for coord in valid_coordinates) for axis in range(3)]
#print(min_coords)
#print(max_coords)
# Calculate the total length in each dimension
total_length_x = max_coords[0] - min_coords[0]
#print(max_coords[1])
#print(min_coords[0])
#print(total_length_x)
total_length_y = max_coords[1] - min_coords[1]
total_length_z = max_coords[2] - min_coords[2]

#print(total_length_x)
#print(total_length_y)
#print(total_length_z)
# Determine the segment length for each dimension
mid_length_x = (total_length_x /  2) + min_coords[0]
mid_length_y = (total_length_y /  2) + min_coords[1]
#mid_length_z = total_length_z /  2
# Initialize the segments
#print(mid_length_x)
#print(mid_length_y)
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

    

# Replace the old subgraph creation with the new segments
G1 = G.subgraph(set(segments[0]))
G2 = G.subgraph(set(segments[1]))
G3 = G.subgraph(set(segments[2]))
G4 = G.subgraph(set(segments[3]))


# Generate all permutations of the elements


#print(G1)
#print(G2)
#print(G3)
#print(G4)

'''
# Visualization function
def draw_graph(G, title):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1000, edge_color='gray')
    plt.title(title)
    plt.show()

# Draw the main graph
draw_graph(G, "Main Graph")

# Draw the subgraphs
draw_graph(G1, "Subgraph  1")
draw_graph(G2, "Subgraph  2")
draw_graph(G3, "Subgraph  3")
draw_graph(G4, "Subgraph  4")
'''

# Generate all permutations of the nodes for each part
permutations1 = list(itertools.permutations(G1.nodes()))
permutations2 = list(itertools.permutations(G2.nodes()))
permutations3 = list(itertools.permutations(G3.nodes()))
permutations4 = list(itertools.permutations(G4.nodes()))

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

print(distances1)
print(distances2)
print(distances3)
print(distances4)

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
            print(last_node_current, first_node_next, connection_weight)
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
print(f"le sortie: {sorted_tuple}")



# Extract the first  10 elements from sorted_tuple
top_ten_elements = sorted_tuple[:10]
#print(f"i top {top_ten_elements}")

# Define the headers for the CSV file
column_names = ['Tour', 'Total Distance']

# Specify the filename
filename = 'top_ten_paths.csv'
#print(enumerate(top_ten_elements))

# Assuming tuplas_totalone is already sorted and contains the top ten tuples
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Path', 'Total Distance'])  # Write the header
    
    # Extract the top ten tuples from tuplas_totalone
    top_ten_tuples = [t[0] for t in sorted_tuple[:10]]
    
    # Write the top ten tuples to the CSV file
    for i, (path_list, distance) in enumerate(top_ten_tuples, start=1):
        # Convert the path list to a string for CSV
        path_str = ', '.join(map(str, path_list))
        csvwriter.writerow([path_str, distance])
      



