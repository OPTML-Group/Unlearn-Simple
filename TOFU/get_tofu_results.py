import os
import pandas as pd

def aggregate_metrics_to_csv_pandas(base_path, steps):
    # Initialize an empty DataFrame
    aggregated_df = pd.DataFrame()
    output_csv=f"aggregated_metrics_pandas_{steps}.csv"
    # Walk through the directories to find the relevant aggregate_stat.txt files
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            subdir_path = os.path.join(root, dir_name, f"checkpoint-{steps}")
            # subdir_path = os.path.join(root, dir_name, "checkpoint-62")
            aggregate_file = os.path.join(subdir_path, "aggregate_stat.txt")
            
            if os.path.isfile(aggregate_file):
                # Read the metrics from the file into a dictionary
                metrics = {}
                with open(aggregate_file, "r") as file:
                    for line in file:
                        key, value = line.strip().split(": ")
                        metrics[key] = float(value) if value.replace('.', '', 1).isdigit() else value

                # Convert the dictionary to a DataFrame and append it to the main DataFrame
                temp_df = pd.DataFrame(metrics, index=[dir_name])
                aggregated_df = pd.concat([aggregated_df, temp_df])

    # Write the aggregated DataFrame to a CSV file
    aggregated_df.to_csv(output_csv)

    print(f"Aggregated data written to {output_csv}")

# Example usage
aggregate_metrics_to_csv_pandas("./paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_full_seed42_1/checkpoint-625/unlearned/", 62)
aggregate_metrics_to_csv_pandas("./paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_full_seed42_1/checkpoint-625/unlearned/", 125)