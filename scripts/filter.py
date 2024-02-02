import json
from create_averages import calculate_averages


def remove_fields_and_save_metrics(input_file_path, output_file_path):
    # Load the original JSON data
    with open(input_file_path, 'r') as file:
        data = json.load(file)
    
    # Remove specified fields from each record
    for record in data:
        if 'precision_recall_curve' in record:
            del record['precision_recall_curve']
        if 'roc_curve' in record:
            del record['roc_curve']
    
    # Save the modified data to a new JSON file
    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=4)
    
    print(f"Modified data saved to {output_file_path}")



def filter_and_save_metrics(input_file_path, output_file_path):
    # Load the original JSON data
    with open(input_file_path, 'r') as file:
        data = json.load(file)
    
    # Create a new data structure for the specified fields
    filtered_data = []
    for record in data:
        filtered_record = {
            'precision_recall_curve': record['precision_recall_curve'],
            'roc_curve': record['roc_curve'],
            'family': record['family'],
            'source': record['source'],
        }
        filtered_data.append(filtered_record)
    
    # Save the filtered data to a new JSON file
    with open(output_file_path, 'w') as file:
        json.dump(filtered_data, file, indent=4)
    
    print(f"Filtered data saved to {output_file_path}")


# deep iterarate through all the files in the baseline_results directory
import os
from pathlib import Path
from numpyencoder import NumpyEncoder
average_metrics = []
for root, dirs, files in os.walk("./result/ojha2022_linear", topdown=False):
    for name in files:
        if name == "metrics.json":
            print(os.path.join(root, name))
            file_path = os.path.join(root, name)
            for family in [None, 'gan', 'diffusion']:
                for source in [None, 'synthbuster']:	
                    ap, roc_auc, threshold_05, oracle = calculate_averages(file_path, family=family, source=source)     
                    if ap is None:
                        continue
                    obj = { 'ap': ap, 'roc_auc': roc_auc, 'threshold_05': threshold_05,
                        'oracle_threshold': oracle, 'family': family if family else 'All',
                        'source': source if source else 'All', 'key': Path(file_path).parent.name}
                    average_metrics.append(obj)

output_file = "./result/ojha2022_linear/average_metrics.json"
with open(output_file, 'w') as file:
    json.dump(average_metrics, file, indent=4, cls=NumpyEncoder)