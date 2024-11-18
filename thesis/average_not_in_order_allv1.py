import csv
import matplotlib.pyplot as plt
import os

def count_non_increasing(sequence):
    non_increasing_count = 0
    prev_value = sequence[0]

    for value in sequence[1:]:
        if value < prev_value:
            non_increasing_count += 1
        prev_value = value

    return non_increasing_count

# Specify the directory containing the CSV files
csv_directory = "/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/grading_trajectories"
# Specify the directory where the plots will be saved
output_directory = "/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/plot_test_grading"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

all_sequences = []

for filename in os.listdir(csv_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_directory, filename)
        sequences = []

        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader) # Skip header row
            sequence_column_index = -1 # Assuming the last column contains the float values

            for row in reader:
                sequence_str = row[sequence_column_index].strip()
                if sequence_str:
                    sequence = list(map(float, sequence_str.split(',')))
                    sequences.append(sequence)

        # Check if the sequence has more than one element before counting non-increasing values
        if len(sequences) > 1:
            non_increasing_count = count_non_increasing(sequences)
            print(f"Number of times values are not in increasing order in {filename}:", non_increasing_count)
        else:
            print(f"Skipping computation for {filename} due to insufficient data.")

        plt.figure()
        #print(sequences)
        plt.plot(sequences)
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title(f"Input Sequences from {filename}")
        plt.savefig(os.path.join(output_directory, filename.replace('.csv', '.png')))
        plt.close()
        
        # Collect all sequences for the final plot
        all_sequences.append(sequences)
        
# Plot all sequences together
plt.figure()
for i, sequence_list in enumerate(all_sequences):
            plt.plot(sequence_list, label=f"Sequence {i+1}")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("All Input Sequences")
plt.legend()
plt.show()
plt.savefig(os.path.join(output_directory, "all_sequences.png"))
plt.close()


