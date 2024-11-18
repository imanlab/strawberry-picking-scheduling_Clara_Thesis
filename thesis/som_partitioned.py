import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def ensure_directory_exists(folder_path):
    """Ensure the directory exists, and if not, create it."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def add_background_image(ax, image_path):
    """Add background image to matplotlib axis."""
    try:
        image = plt.imread(image_path)
        im = OffsetImage(image, zoom=1)
        ab = AnnotationBbox(im, (0.5, 0.5), frameon=False, xycoords='axes fraction', boxcoords="axes fraction")
        ax.add_artist(ab)
    except Exception as e:
        print(f"Failed to load background image: {e}")

def plot_network(cities, neurons, name='diagram.png', image_path=None, ax=None):
    """Plot a graphical representation of the problem with optional background image."""
    mpl.rcParams['agg.path.chunksize'] = 10000
    ensure_directory_exists(os.path.dirname(name))  # Ensure the directory exists before saving
    cities = cities.dropna(subset=['z'])

    if not ax:
        fig = plt.figure(figsize=(5, 5), frameon=False)
        axis = fig.add_axes([0, 0, 1, 1])
        axis.set_aspect('equal', adjustable='datalim')
        plt.axis('off')

        if image_path:
            add_background_image(axis, image_path)

        axis.scatter(cities['x'], cities['y'], color='red', s=4)
        axis.plot(neurons[:, 0], neurons[:, 1], color='#0063ba', markersize=2)
        plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
    else:
        if image_path:
            add_background_image(ax, image_path)
        ax.scatter(cities['x'], cities['y'], color='red', s=4)
        ax.plot(neurons[:, 0], neurons[:, 1], color='#0063ba', markersize=2)
        return ax

def plot_route(cities, route, name='diagram.png', image_path=None, ax=None):
    """Plot a graphical representation of the route with optional background image."""
    mpl.rcParams['agg.path.chunksize'] = 10000
    ensure_directory_exists(os.path.dirname(name))  # Ensure the directory exists before saving
    cities = cities.dropna(subset=['z'])
    route = cities.reindex(route)

    if not ax:
        fig = plt.figure(figsize=(5, 5), frameon=False)
        axis = fig.add_axes([0, 0, 1, 1])
        axis.set_aspect('equal', adjustable='datalim')
        plt.axis('off')

        if image_path:
            add_background_image(axis, image_path)

        axis.scatter(cities['x'], cities['y'], color='red', s=4)
        if not route.empty:
            route.loc[route.shape[0]] = route.iloc[0]  # Close the loop
            axis.plot(route['x'], route['y'], color='purple', linewidth=1)

        plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
    else:
        if image_path:
            add_background_image(ax, image_path)
        ax.scatter(cities['x'], cities['y'], color='red', s=4)
        if not route.empty:
            route.loc[route.shape[0]] = route.iloc[0]  # Close the loop
            ax.plot(route['x'], route['y'], color='purple', linewidth=1)
        return ax

import numpy as np
import pandas as pd

def select_closest(candidates, origin):
    """Return the index of the closest candidate to a given point."""
    return euclidean_distance(candidates, origin).argmin()

def euclidean_distance(a, b):
    """Return the array of distances of two numpy arrays of points."""
    return np.linalg.norm(a - b, axis=1)

def route_distance(cities, alpha=0.1):
    """
    Return the cost of traversing a route of cities in a certain order, including 3D coordinates.
    Calculates the route only until the last city, without returning to the start.
    Incorporates hardness of picking as a weighted distance.
    """
    # Filter out any rows where 'z' is NaN before calculating distances
    filtered_cities = cities.dropna(subset=['z'])

    if filtered_cities.empty:
        return 0  # Return a distance of 0 if there are no valid cities to process

    # Calculate the distance between consecutive cities
    points = filtered_cities[['x', 'y', 'z']].to_numpy()
    hardness = filtered_cities['Hardness of Picking'].to_numpy().astype(float)
    distances = euclidean_distance(points[:-1], points[1:])

    # Incorporate hardness of picking into the distance
    weighted_distances = distances * (1 + alpha * hardness[1:])
    return np.sum(weighted_distances)
import pandas as pd
import numpy as np
import csv

def read_csv_as_tsp(filename):
    """
    Read a CSV file with 3D coordinates and hardness of picking into a pandas DataFrame, handling missing z-coordinates.
    Each row in the CSV file should represent a city with its x, y, z coordinates, and hardness of picking.
    The DataFrame will include an index column to identify each node.
    """
    coordinates = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for index, row in enumerate(reader):
            if row:  # Skip empty rows
                try:
                    x = float(row[0])
                    y = float(row[1])
                    z = float(row[2]) if len(row) > 2 and row[2] else None  # Handle missing z by setting as NaN
                    hardness_of_picking = int(row[3]) if len(row) > 3 and row[3] else None  # Handle missing hardness by setting as None
                    coordinates.append((index, x, y, z, hardness_of_picking))
                except ValueError:
                    continue  # Skip rows with invalid data

    # Create a DataFrame from the list of tuples
    cities = pd.DataFrame(coordinates, columns=['index', 'x', 'y', 'z', 'Hardness of Picking'])
    print('Data with {} points read.'.format(len(cities)))
    #print(cities)
    return cities

def normalize_3d(points):
    """
    Normalize the 3D coordinates. This involves:
    - Offsetting each dimension by its minimum value.
    - Scaling each dimension independently to the range [0,1].
    Note: Handles NaN in the z-coordinate by ignoring them in normalization.
    """
    for column in ['x', 'y', 'z']:
        if points[column].notna().any():
            points.loc[:, column] = (points[column] - points[column].min()) / (points[column].max() - points[column].min())
            #print(f"Normalized {column}:")
            #print(points[column].head())  # Print the first few normalized values for the column
    return points
import numpy as np



def generate_network(size):
    """
    Generate a neuron network of a given size.

    Return a vector of three-dimensional points in the interval [0,1].
    """
    return np.random.rand(size, 3)

def get_neighborhood(center, radix, domain):
    """Get the range gaussian of given radix around a center index."""

    # Impose an upper bound on the radix to prevent NaN and blocks
    if radix < 1:
        radix = 1

    # Compute the circular network distance to the center
    deltas = np.absolute(center - np.arange(domain))
    distances = np.minimum(deltas, domain - deltas)

    # Compute Gaussian distribution around the given center
    return np.exp(-(distances**2) / (2*(radix**2)))


def get_route(cities, network):
    """
    Return the route computed by a network. Assumes that cities DataFrame includes
    an 'index' along with 'x', 'y', 'z' coordinates.
    Cities with 'NaN' in the 'z' coordinate are skipped.
    """
    # Filter out any cities where 'z' is NaN before processing
    valid_cities = cities.dropna(subset=['z'])

    # Calculate the closest network node ('winner') for each valid city
    valid_cities['winner'] = valid_cities[['x', 'y', 'z']].apply(
        lambda row: select_closest(network, row.values.reshape(1, -1)), axis=1
    )

    # Sort valid cities by the 'winner' neuron to establish the route order
    sorted_cities = valid_cities.sort_values('winner')

    # Return the index of sorted cities, preserving the original DataFrame indexing
    return sorted_cities['index']


import pandas as pd
import numpy as np
import csv
import os

def process_directory(directory):
    if not os.path.isdir(directory):
        print(f"The specified path is not a directory: {directory}")
        return -1

    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in the directory: {directory}")
        return

    # Create an output directory within the specified directory
    output_dir = os.path.join(directory, "output")
    os.makedirs(output_dir, exist_ok=True)

    alpha_values = np.arange(0, 10.1, 0.1)  # Alpha values from 0.1 to 10, increasing by 0.1

    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)
        try:
            problem = read_csv_as_tsp(file_path)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")
            continue

        # Filter out entries with None in 'z'
        problem = problem.dropna(subset=['z'])
        if problem.empty:
            print(f"No valid data to process after filtering out incomplete entries in {file_path}.")
            continue

        for alpha in alpha_values:
            results = []
            diagram_dir = os.path.join(output_dir, f"diagrams_alpha_{alpha:.1f}_{os.path.splitext(csv_file)[0]}")
            os.makedirs(diagram_dir, exist_ok=True)

            for i in range(10):
                route = som(problem, 100000, alpha=alpha, diagram_dir=diagram_dir, iteration=i)

                # Reindex the problem DataFrame according to the computed route and then reset index to get the ordered indices
                ordered_problem = problem.set_index('index').reindex(route).reset_index()
                distance = route_distance(ordered_problem, alpha=alpha)

                # Extract the hardness of picking values corresponding to the ordered path
                hardness_values = ordered_problem['Hardness of Picking'].tolist()

                # Append the result to the results list
                results.append([alpha, ordered_problem['index'].tolist(), hardness_values, distance])

            # Save results to a new CSV file in the output directory
            output_file_path = os.path.join(output_dir, f"result_alpha_{alpha:.1f}_{os.path.basename(csv_file)}")
            with open(output_file_path, 'w', newline='') as output_file:
                writer = csv.writer(output_file)
                writer.writerow(['Alpha', 'Path', 'Hardness of Picking', 'Total Length'])
                writer.writerows(results)

            print(f"Processed {csv_file} with alpha={alpha:.1f}: saved 10 results to {output_file_path}")

def main():
    # Hardcoded directories containing the CSV files
    
    directory2 = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/coordinates_and_hop_6'  # Change this to the path to your second folder of CSV files

   
    process_directory(directory2)

def som(problem, iterations, learning_rate=0.8, alpha=0.1, diagram_dir=None, iteration=0):
    """Solve the TSP using a Self-Organizing Map."""
    cities = problem.copy()
    cities.set_index('index', inplace=True)  # Use index from DataFrame if needed for reordering
    cities[['x', 'y', 'z']] = normalize_3d(cities[['x', 'y', 'z']])

    n = cities.shape[0] * 8  # The population size is 8 times the number of cities
    network = generate_network(n)
    print(f'Network of {n} neurons created. Starting the iterations for alpha={alpha}:')

    for i in range(iterations):
        if not i % 100:
            print(f'\t> Iteration {i}/{iterations} for alpha={alpha}', end="\r")
        city = cities.sample(1)[['x', 'y', 'z']].values
        winner_idx = select_closest(network, city)
        gaussian = get_neighborhood(winner_idx, n//10, network.shape[0])
        network += gaussian[:, np.newaxis] * learning_rate * (city - network)
        learning_rate = learning_rate * 0.99997
        n = n * 0.9997

        if diagram_dir and not i % 1000:
            plot_network(cities.reset_index(), network, name=os.path.join(diagram_dir, f'{iteration}_{i:05d}.png'))

        if n < 1 or learning_rate < 0.001:
            print(f'Finishing execution at {i} iterations for alpha={alpha}')
            break
    else:
        print(f'Completed {iterations} iterations for alpha={alpha}')

    if diagram_dir:
        plot_network(cities.reset_index(), network, name=os.path.join(diagram_dir, f'{iteration}_final.png'))
    route = get_route(cities.reset_index(), network)
    if diagram_dir:
        plot_route(cities.reset_index(), route, name=os.path.join(diagram_dir, f'{iteration}_route.png'))
    return route.tolist()

if __name__ == '__main__':
    main()

