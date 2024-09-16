import os
import json
import pandas as pd
from datetime import datetime

def get_latest_json_files(folder_path, n=3):
    """Get the latest `n` JSON files from a given folder."""
    json_files = []
    
    # Traverse through the folder and list all JSON files
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json") and file_name.startswith("results_"):
            file_path = os.path.join(folder_path, file_name)
            # Extract the datetime from the filename and store along with the path
            date_str = file_name.replace("results_", "").replace(".json", "")
            date_obj = datetime.strptime(date_str, "%Y-%m-%dT%H-%M-%S.%f")
            json_files.append((date_obj, file_path))
    
    # Sort by datetime and return the latest `n` files
    json_files.sort(reverse=True, key=lambda x: x[0])
    return [file[1] for file in json_files[:n]]

def extract_data_from_json(file_path):
    """Extract the required data from a JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    extracted_data = {}
    if "wmdp_bio" in data["results"]:
        extracted_data["wmdp_bio"] = 1.0 - data["results"]["wmdp_bio"].get("acc,none")
    if "wmdp_cyber" in data["results"]:
        extracted_data["wmdp_cyber"] = 1.0 - data["results"]["wmdp_cyber"].get("acc,none")
    if "mmlu" in data["results"]:
        extracted_data["mmlu"] = data["results"]["mmlu"].get("acc,none")
    
    return extracted_data

def main(root_folder_path, output_csv_path):
    """Main function to generate the CSV from JSON files in the subfolders."""
    result_dict = {}
    
    for subfolder_name in os.listdir(root_folder_path):
        subfolder_path = os.path.join(root_folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            latest_files = get_latest_json_files(subfolder_path)
            combined_data = {"wmdp_bio": None, "wmdp_cyber": None, "mmlu": None}
            
            for file in latest_files:
                extracted_data = extract_data_from_json(file)
                for key in combined_data:
                    if extracted_data.get(key) is not None:
                        combined_data[key] = extracted_data[key]
            
            result_dict[subfolder_name] = combined_data
    
    # Convert the result dictionary to a DataFrame
    df = pd.DataFrame.from_dict(result_dict, orient='index')
    
    # Save to CSV
    df.to_csv(output_csv_path)

# Example usage
root_folder_path = "/egr/research-optml/chongyu/NEW-BLUE/TOFU/wmdp_models/result"  # Replace with the path to your root folder
output_csv_path = "/egr/research-optml/chongyu/NEW-BLUE/TOFU/wmdp_models/wmdp_result.csv"  # Replace with the desired output CSV file path

main(root_folder_path, output_csv_path)