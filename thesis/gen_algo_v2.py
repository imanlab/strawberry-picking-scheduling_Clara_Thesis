import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import csv
import os
import itertools
from more_itertools import pairwise
from itertools import permutations
import glob
import re
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from random import shuffle
import statistics
import seaborn as sns

def write_to_csv(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

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
    
def create_graph(coordinates):
   # Initialize the graph
   G = nx.Graph()

   #Add all points as nodes to the graph using the original index as the node label
   for index, coord in coordinates:
      #print(index)
      #print(coord)
      if coord[2] is not None:  # Only add nodes with a valid z-coordinate
          G.add_node(index, pos=coord)
          #print(index)
          #print(coord)

   # Add edges to the graph
   for u, v in itertools.combinations(G.nodes,  2):
      if u != v:
          G.add_edge(u, v, weight=np.linalg.norm(np.array(G.nodes[u]['pos']) - np.array(G.nodes[v]['pos'])))
          
   #print(G.nodes)
   
   return G
 
def initial_population(cities_names, n_population=250):
    """
    Generating initial population of nodes randomly selected from all   
    possible permutations of the given nodes
    Input:
    1- File path for coordinates
    2. Number of population   
    Output:
    Generated lists of nodes
    """
    coordinates = get_coordinates(file_path)
    G = create_graph(coordinates)
    # Get the list of nodes (indices) from the graph
    cities_names = list(G.nodes)
    
    population_perms = []
    for _ in range(n_population):
        shuffled_cities = cities_names.copy()  # Create a copy to avoid modifying the original list
        
        random.shuffle(shuffled_cities)  # Shuffle the copy to get a random permutation
        
        population_perms.append(shuffled_cities)
        
    return population_perms
    
def dist_two_nodes(coordinates, node_1, node_2):
    """
    Returning the weight of the edge between two nodes
    Input:
    1- Coordinates list
    2- Node one name
    3- Node two name
    Output:
    Weight of the edge between the two nodes
    """

    # Create the graph using the provided coordinates
    G = create_graph(coordinates)

    if node_1 in G.nodes and node_2 in G.nodes:
        # Now it's safe to call dist_two_nodes
        return G[node_1][node_2]['weight']
    else:
        print(f"Node(s) not found in graph: {node_1}, {node_2}")
        return None 
    
def total_dist_individual(individual):

    
    """
    Calculating the total distance traveled by individual, 
    one individual means one possible solution (1 permutation)
    Input:
    1- Individual list of cities 
    Output:
    Total distance traveled 
    """
    # compare with brute force
    

    total_dist =  0
    for i in range(len(individual) -  1):  # Adjust the range to not include the last index
        total_dist += dist_two_nodes(coordinates, individual[i], individual[i+1])
            #print(total_dist)
    # Add the distance from the last city to the first city
    total_dist += dist_two_nodes(coordinates, individual[-1], individual[0])
    return total_dist

def fitness_prob(population):

    """
    Calculating the fitness probability 
    Input:
    1- Population  
    Output:
    Population fitness probability 
    """
    
    total_dist_all_individuals = []
    
    for i in range (0, len(population)):
        total_dist_all_individuals.append(total_dist_individual(population[i]))
        
    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - total_dist_all_individuals
    population_fitness_sum = sum(population_fitness)
    population_fitness_probs = population_fitness / population_fitness_sum
    
    return population_fitness_probs
  
def roulette_wheel(population, fitness_probs):

    """
    Implement a selection strategy based on proportionate roulette wheel
    Selection.
    Input:
    1- population
    2: fitness probabilities 
    Output:
    selected individual
    """
    population_fitness_probs_cumsum = fitness_probs.cumsum()
    bool_prob_array = population_fitness_probs_cumsum < np.random.uniform(0,1,1)
    selected_individual_index = len(bool_prob_array[bool_prob_array == True]) - 1
    return population[selected_individual_index]

def crossover(parent_1, parent_2):

    """
    Implement mating strategy using simple crossover between two parents
    Input:
    1- parent 1
    2- parent 2 
    Output:
    1- offspring 1
    2- offspring 2
    """
    
    n_cities_cut = len(cities_names) - 1
    cut = round(random.uniform(1, n_cities_cut))
    offspring_1 = []
    offspring_2 = []
    
    offspring_1 = parent_1 [0:cut]
    offspring_1 += [city for city in parent_2 if city not in offspring_1]
    
    
    offspring_2 = parent_2 [0:cut]
    offspring_2 += [city for city in parent_1 if city not in offspring_2]
    
    
    return offspring_1, offspring_2
    
def mutation(offspring):

    """
    Implement mutation strategy in a single offspring
    Input:
    1- offspring individual
    Output:
    1- mutated offspring individual
    """
    
    n_cities_cut = len(cities_names) - 1
    index_1 = round(random.uniform(0,n_cities_cut))
    index_2 = round(random.uniform(0,n_cities_cut))

    temp = offspring [index_1]
    offspring[index_1] = offspring[index_2]
    offspring[index_2] = temp
    
    return(offspring)
    
    
def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
    
    population = initial_population(cities_names, n_population)
    #print(population)
    fitness_probs = fitness_prob(population)
    
    parents_list = []
    for i in range(0, int(crossover_per * n_population)):
        parents_list.append(roulette_wheel(population, fitness_probs))

    offspring_list = []    
    for i in range(0,len(parents_list), 2):
        offspring_1, offspring_2 = crossover(parents_list[i], parents_list[i+1])

        #print(offspring_1)
        #print(offspring_2)
        #print()

        mutate_threashold = random.random()
        if(mutate_threashold > (1-mutation_per)):
            offspring_1 = mutation(offspring_1)
            print("Offspring 1 mutated", offspring_1)

        mutate_threashold = random.random()
        if(mutate_threashold > (1-mutation_per)):
            offspring_2 = mutation(offspring_2)
            print("Offspring 2 mutated", offspring_2)


        offspring_list.append(offspring_1)
        offspring_list.append(offspring_2)

    mixed_offspring = parents_list + offspring_list

    fitness_probs = fitness_prob(mixed_offspring)
    sorted_fitness_indices = np.argsort(fitness_probs)[::-1]
    best_fitness_indices = sorted_fitness_indices[0:n_population]
    best_mixed_offspring = []
    for i in best_fitness_indices:
        best_mixed_offspring.append(mixed_offspring[i])
        
    # List to store total distances for each generation
    individual_distances = [] 
    data=[]   

    for i in range(0, n_generations):
        if (i%10 == 0):
             print("Generation: ", i)
        
        fitness_probs = fitness_prob(best_mixed_offspring)
       
        
        # After calculating the fitness probabilities, write the data to the CSV file
        for individual in best_mixed_offspring:
            total_dist = total_dist_individual(individual)
            for idx in range(len(individual)):
                if idx < len(individual) -  1:
                    node1 = individual[idx]
                    node2 = individual[idx +  1]
                    weight = dist_two_nodes(coordinates, node1, node2)
                    data.append([i, node1, node2, weight, total_dist])
                else:  # For the last node, connect it back to the first node
                    node1 = individual[idx]
                    node2 = individual[0]
                    weight = dist_two_nodes(coordinates, node1, node2)
                    data.append([i, node1, node2, weight, total_dist])

        # Write the data to the CSV file after each generation
        write_to_csv(data, 'weights_and_distances.csv')

        # Clear the data list for the next generation
        data = []
        
        parents_list = []
        for i in range(0, int(crossover_per * n_population)):
            parents_list.append(roulette_wheel(best_mixed_offspring, fitness_probs))

        offspring_list = []    
        for i in range(0,len(parents_list), 2):
            offspring_1, offspring_2 = crossover(parents_list[i], parents_list[i+1])

            mutate_threashold = random.random()
            if(mutate_threashold > (1-mutation_per)):
                offspring_1 = mutation(offspring_1)

            mutate_threashold = random.random()
            if(mutate_threashold > (1-mutation_per)):
                offspring_2 = mutation(offspring_2)

            offspring_list.append(offspring_1)
            offspring_list.append(offspring_2)


        mixed_offspring = parents_list + offspring_list
        fitness_probs = fitness_prob(mixed_offspring)
        
        sorted_fitness_indices = np.argsort(fitness_probs)[::-1]
        best_fitness_indices = sorted_fitness_indices[0:int(0.8*n_population)]

        best_mixed_offspring = []
        for i in best_fitness_indices:
            best_mixed_offspring.append(mixed_offspring[i])
            
        old_population_indices = [random.randint(0, (n_population - 1)) for j in range(int(0.2*n_population))]
        for i in old_population_indices:
            #print(i)
            best_mixed_offspring.append(population[i])
            
        random.shuffle(best_mixed_offspring)
        
        '''
        # After each generation, calculate and store the total distance
        total_distance = sum([total_dist_individual(individual) for individual in population])
        total_distances.append(total_distance)
        '''
        
        # Calculate the total distance for each individual in the best_mixed_offspring
        total_distances = [total_dist_individual(individual) for individual in best_mixed_offspring]
        # Append the total distances for this generation to the list
        individual_distances.append(total_distances)    
        
    # Create a dictionary to store the paths and their distances
    paths_and_distances = {}

    for individual in best_mixed_offspring:
        total_dist = total_dist_individual(individual)
        paths_and_distances[tuple(individual)] = total_dist
        

    return paths_and_distances, individual_distances

file_path ='/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_dataset-master/scheduling_dataset-main/scheduling_riseholme/train/coordinates_for_images/102.csv'    
coordinates = get_coordinates(file_path)
# Create the graph
G = create_graph(coordinates)
# Get the list of nodes (indices) from the graph
cities_names = list(G.nodes)
print(cities_names)
n_population = 300
crossover_per = 0.9
mutation_per = 0.1
n_generations = 100

paths_and_distances, individual_distances = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)

# Get the top ten paths and their distances
top_ten_paths = sorted(paths_and_distances.items(), key=lambda x: x[1])[:10]

# Now you can access the paths and distances using the index
for i in range(len(top_ten_paths)):
    path, distance = top_ten_paths[i]
    print(f"Path: {path}, Distance: {distance}")
    

# Plotting the total_dist_individual for every ten generations
for generation, distances in enumerate(individual_distances):
    if generation %  10 ==  0:  # Check if the generation number is a multiple of  10
        plt.plot(distances, label=f'Generation {generation}')

plt.xlabel('Individual Index')
plt.ylabel('Total Distance')
plt.title('Total Distance for eac individual Over 10 Generations')
plt.legend()
plt.show()
  
