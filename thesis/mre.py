import os
import pandas as pd

def compute_average_errors(input_directory):
    # Create an empty DataFrame to store the data
    df = pd.DataFrame(columns=["Ground Truth File", "Predicted File", "Mean Total Length Error", "Mean Scheduling Order Error"])

    # Iterate over all CSV files in the input directory
    for file_name in os.listdir(input_directory):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_directory, file_name)
            temp_df = pd.read_csv(file_path)
            df = pd.concat([df, temp_df], ignore_index=True)
    
    # Compute the average errors
    avg_total_length_error = df["Mean Total Length Error"].mean()
    avg_scheduling_order_error = df["Mean Scheduling Order Error"].mean()

    print(f"Average Total Length Error: {avg_total_length_error}")
    print(f"Average Scheduling Order Error: {avg_scheduling_order_error}")

    return avg_total_length_error, avg_scheduling_order_error

# Example usage
input_directory = "/home/clara/Thesis/strawberry-pp-w-r-dataset-master/compa_paths_k_g"
compute_average_errors(input_directory)

