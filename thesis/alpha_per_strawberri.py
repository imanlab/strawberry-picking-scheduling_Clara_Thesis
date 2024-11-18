import pandas as pd
import matplotlib.pyplot as plt

def plot_alpha_vs_nodes(csv_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Ensure the required columns are present
    if 'Best Alpha' not in df.columns or 'Number of Nodes' not in df.columns:
        raise ValueError("CSV file must contain 'Best Alpha' and 'Number of Nodes' columns")

    # Sort the DataFrame by 'Number of Nodes' to make the plot more readable
    df = df.sort_values(by='Number of Nodes')
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df['Number of Nodes'], df['Best Alpha'], marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Best Alpha')
    plt.title('Best Alpha vs Number of Nodes')
    plt.grid(True)
    plt.show()

# Usage example
csv_file_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/results.csv'
plot_alpha_vs_nodes(csv_file_path)

