import json
import argparse
from numpyencoder import NumpyEncoder

def calculate_averages(file_path, family=None, source=None):
    # Load the JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    if family is not None:
        data = [record for record in data if record['family'] == family]

    if source is not None:
        data = [record for record in data if record['source'] == source]

    if len(data) == 0:
        return None, None, None, None
    
    # Initialize sums and counters for the fields
    sum_ap = 0
    sum_roc_auc = 0
    sum_threshold_05 = {key: 0 for key in data[0]['threshold_05'].keys()}
    sum_oracle_threshold = {key: 0 for key in data[0]['oracle_threshold'].keys()}
    count = len(data)
    
    # Iterate through each record to sum values
    for record in data:
        sum_ap += record['ap']
        sum_roc_auc += record['roc_auc']
        for key in sum_threshold_05.keys():
            sum_threshold_05[key] += record['threshold_05'][key]
        for key in sum_oracle_threshold.keys():
            sum_oracle_threshold[key] += record['oracle_threshold'][key]
    
    # Calculate averages
    avg_ap = sum_ap / count
    avg_roc_auc = sum_roc_auc / count
    avg_threshold_05 = {key: value / count for key, value in sum_threshold_05.items()}
    avg_oracle_threshold = {key: value / count for key, value in sum_oracle_threshold.items()}
    
    # Return the averages
    return avg_ap, avg_roc_auc, avg_threshold_05, avg_oracle_threshold


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--family', type=str, default=None)
    parser.add_argument('--source', type=str, default=None)

    opt = parser.parse_args()

    averages = calculate_averages(opt.input_file, opt.family, opt.source)

    print(f"Average AP: {averages[0]}")
    print(f"Average ROC AUC: {averages[1]}")
    print(f"Average Threshold 0.5: {averages[2]}")
    print(f"Average Oracle Threshold: {averages[3]}")

    with open(opt.output_file, 'w') as file:
        obj = {
            'ap': averages[0],
            'roc_auc': averages[1],
            'threshold_05': averages[2],
            'oracle_threshold': averages[3],
            'family': opt.family if opt.family else 'All',
            'source': opt.source if opt.source else 'All'
        }
        json.dump(obj, file, indent=4, cls=NumpyEncoder)

