import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def count_objects_in_files(folder_paths):
    # Dictionary to store the count of objects for each file
    object_counts = {}
    
    for folder_path in folder_paths:
        # List all CSV files in the folder
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            
            # Count the rows where the third column is not empty
            count = df.dropna(subset=[df.columns[2]], how='all').shape[0]
            
            # Store the count in the dictionary
            object_counts[file] = count
    
    return object_counts

def plot_histogram(object_counts):
    # Extract the counts
    counts = list(object_counts.values())
    
    # Set the style and color palette of the plot
    sns.set(style="whitegrid")
    
    # Create a histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(counts, bins=range(min(counts), max(counts) + 2), kde=False, color='skyblue', edgecolor='black')
    
    # Add titles and labels
    plt.title('Number of strawberries per image', fontsize=16)
    plt.xlabel('Number of strawberries', fontsize=14)
    plt.ylabel('Number of images', fontsize=14)
    
    # Customize the ticks on x and y axes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Display the histogram
    plt.show()

# Example usage
folder_paths = ['/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/coordinates_for_images', '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/test/coordinates_for_images', '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/val/coordinates_for_images', '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/coordinates_for_images', '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/test/coordinates_for_images', '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/val/coordinates_for_images']  # Replace with your folder paths
object_counts = count_objects_in_files(folder_paths)

# Print the object counts
for file, count in object_counts.items():
    print(f'File: {file}, Number of Objects: {count}')

# Plot the histogram
plot_histogram(object_counts)

