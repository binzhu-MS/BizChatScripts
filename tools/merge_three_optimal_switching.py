# read three JSON files and merge them into one

import json
import os
import sys
import re

Merged_Optimal_Switching = {
    "ReasoningClasses": []
}

def merge_json_files(file_paths):
    merged_data = {}

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            continue

        with open(file_path, 'r') as file:
            data = json.load(file)
            classes = data["CITEDCG_ONE_CENTRIC"]["ReasoningClasses"]
            merged_data["ReasoningClasses"] = list(set(merged_data.get("ReasoningClasses", []) + classes))

    return merged_data


file_paths = ["107014_Optimal_Switching.json", "107015_Optimal_Switching.json", "107013_Optimal_Switching.json"]

# save the merged data to a new JSON file
merged_data = merge_json_files(file_paths)
output_file = "Merged_Optimal_Switching.json"
with open(output_file, 'w') as outfile:
    json.dump(merged_data, outfile, indent=2)