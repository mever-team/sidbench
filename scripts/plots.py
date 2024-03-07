import json
import matplotlib.pyplot as plt
import numpy as np
import os

from pathlib import Path


PALLETE = {
    "blue": "#BDE0FE",
    "green": "#A2D2FF",
    "grey": "#A9A9A9",
    "teal": "#AFDBD2",
    "navy": "#005F73",
    "earth_green": "#556B2F",
    "earth_brown": "#8B4513",
    "earth_beige": "#D2B48C",
    "earth_olive": "#6B8E23",
    "earth_rust": "#B7410E",
}


# Attach a text label above each bar in *bars1* and *bars2*, displaying its height.
def autolabel(bars, ax):
    """Attach a text label above each bar in *bars*, displaying its height."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(round(height, 2)),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 2),  # 2 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom')
        

def save_ap_auc_barchart(input_file_path, output_file_path, xlabel, trained_on=None, family=None, source=None):

    # Load the JSON data
    with open(input_file_path, 'r') as file:
        data = json.load(file)

    if family is not None:
        data = [record for record in data if record['family'] == family]

    if source is not None:
        data = [record for record in data if record['source'] == source]

    # Extract ACC, AP and ROC AUC values
    acc_values = [record['threshold_05']['acc'] for record in data]
    ap_values = [record['ap'] for record in data]
    roc_auc_values = [record['roc_auc'] for record in data]
    indices = np.arange(len(data))  # The label locations
    
    # Bar chart parameters
    width = 0.25  # the width of the bars

    # Plot AP and ROC AUC values as bar charts
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(indices - width, acc_values, width, label='Acc', color=PALLETE['blue'])
    ax.bar(indices, ap_values, width, label='AP', color=PALLETE['navy'])
    ax.bar(indices + width, roc_auc_values, width, label='ROC AUC', color=PALLETE['teal'])

    title = '' if trained_on is None else 'Trained on: ' + trained_on + ' / '
    title += ' Test on: '
    if family is None and source is None or family == 'All' and source == 'All':
        title += 'All'
    elif family is not None:
        title += f'{family}'
    
    if source is not None and source != 'All':
        title += f' ({source})'

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)

    ax.set_title(title)
    ax.set_ylabel('AP / ROC AUC')
    ax.set_xlabel(xlabel)
    ax.set_xticks(indices)

    tick_labels = [f'{data[i]["key"]}'  for i in indices]
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
    # ax.set_xticklabels([f'Record {i+1}' for i in indices])
    ax.legend()

    #autolabel(bars1)
    #autolabel(bars2)

    fig.tight_layout()

    # Save the plot to a file
    plt.savefig(output_file_path)
    plt.close()


def save_acc_barchart(input_file_path, output_file_path, xlabel, trained_on=None, family=None, source=None):
    # Load the JSON data
    with open(input_file_path, 'r') as file:
        data = json.load(file)
    
    if family is not None:
        data = [record for record in data if record['family'] == family]
    
    if source is not None:
        data = [record for record in data if record['source'] == source]

    acc_values = [record['threshold_05']['acc'] for record in data]
    oracle_acc_values = [record['oracle_threshold']['acc'] for record in data]
    tpr_values = [record['oracle_threshold']['tpr'] for record in data]
    tnr_values = [record['oracle_threshold']['tnr'] for record in data]
    indices = np.arange(len(data))  # The label locations
    
    # Bar chart parameters
    width = 0.15  # the width of the bars

    # Plot AP and ROC AUC values as bar charts
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(indices - 3*width/2, acc_values, width, label='Accuracy', color=PALLETE['navy'])
    ax.bar(indices - width/2, oracle_acc_values, width, label='Oracle Accuracy', color=PALLETE['teal'])
    ax.bar(indices + width/2, tpr_values, width, label='Oracle TPR', color=PALLETE['grey'])
    ax.bar(indices + 3*width/2, tnr_values, width, label='Oracle TNR', color=PALLETE['blue'])
    
    title = '' if trained_on is None else 'Trained on: ' + trained_on + ' / '
    title += ' Test on: '
    if family is None and source is None or family == 'All' and source == 'All':
        title += 'All'
    elif family is not None:
        title += f'{family}'
    
    if source is not None and source != 'All':
        title += f' ({source})'

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)

    ax.set_title(title)

    ax.set_ylabel('Accuracy')
    ax.set_xlabel(xlabel)
    # ax.set_ylabel('Metric')``
    ax.set_xticks(indices)

    tick_labels = [f'{data[i]["key"]}'  for i in indices]
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
    ax.legend()

    fig.tight_layout()

    # Save the plot to a file
    plt.savefig(output_file_path)
    plt.close()


def save_metrics_barchart(input_file_path, output_file_path, xlabel, trained_on=None, family=None, source=None):
    # Load the JSON data
    with open(input_file_path, 'r') as file:
        data = json.load(file)

    if family is not None:
        data = [record for record in data if record['family'] == family]
    
    if source is not None:
        data = [record for record in data if record['source'] == source]

    # Extract AP and ROC AUC values
    acc_values = [record['threshold_05']['acc'] for record in data]
    tpr_values = [record['threshold_05']['tpr'] for record in data]
    tnr_values = [record['threshold_05']['tnr'] for record in data]
    indices = np.arange(len(data))  # The label locations
    
    # Bar chart parameters
    width = 0.2  # the width of the bars

    # Plot AP and ROC AUC values as bar charts
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(indices - width, acc_values, width, label='Accuracy', color=PALLETE['navy'])
    ax.bar(indices, tpr_values, width, label='TPR (fakes)', color=PALLETE['teal'])
    ax.bar(indices + width, tnr_values, width, label='TNR (reals)', color=PALLETE['grey'])

    title = '' if trained_on is None else 'Trained on: ' + trained_on + ' / '
    title += ' Test on: '
    if family is None and source is None or family == 'All' and source == 'All':
        title += 'All'
    elif family is not None:
        title += f'{family}'
    
    if source is not None and source != 'All':
        title += f' ({source})'


    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)

    ax.set_title(title)

    ax.set_ylabel('Classification Metrics')
    ax.set_xlabel(xlabel)
    ax.set_xticks(indices)

    tick_labels = [f'{data[i]["key"]}'  for i in indices]
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
    ax.legend()

    fig.tight_layout()

    # Save the plot to a file
    plt.savefig(output_file_path)
    plt.close()


input_file_path = './test_results/tan2023_linear/_jpeg_None_gaussian_None/metrics.json'
output_folder = './figs/' + Path(input_file_path).parent.parent.name
print(output_folder)

save_metrics_barchart(input_file_path, output_folder + '/metrics_lgrad.png', 'Dataset')
save_acc_barchart(input_file_path, output_folder + '/oracle_acc_lgrad.png', 'Dataset')
save_ap_auc_barchart(input_file_path, output_folder + '/ap_auc_lgrad.png', 'Dataset')

save_metrics_barchart(input_file_path, output_folder + '/metrics_wang2020_lgrad.png', 'Dataset', family='gan', source='wang2020')
save_acc_barchart(input_file_path, output_folder + '/oracle_acc_wang2020_lgrad.png', 'Dataset', family='gan', source='wang2020')
save_ap_auc_barchart(input_file_path, output_folder + '/ap_auc_wang2020_lgrad.png', 'Dataset', family='gan', source='wang2020')

save_metrics_barchart(input_file_path, output_folder + '/metrics_synthbuster_lgrad.png', 'Dataset', source='synthbuster')
save_acc_barchart(input_file_path, output_folder + '/oracle_acc_synthbuster_lgrad.png', 'Dataset', source='synthbuster')
save_ap_auc_barchart(input_file_path, output_folder + '/ap_auc_synthbuster_lgrad.png', 'Dataset', source='synthbuster')


# for root, dirs, files in os.walk("./baseline_results", topdown=False):
#     for name in files:
#         if name == "metrics.json":
#             print(os.path.join(root, name))
#             input_file_path = os.path.join(root, name)
#             output_folder = os.path.join('./figs', Path(input_file_path).parent.name)
#             if not os.path.exists(output_folder):
#                 os.makedirs(output_folder)
#             print(output_folder)
#             save_metrics_barchart(input_file_path, output_folder + '/metrics.png', 'Dataset')
#             save_acc_barchart(input_file_path, output_folder + '/oracle_acc.png', 'Dataset')
#             save_ap_auc_barchart(input_file_path, output_folder + '/ap_auc.png', 'Dataset')
#             save_metrics_barchart(input_file_path, output_folder + '/metrics_gan.png', 'Dataset', family='gan')
#             save_acc_barchart(input_file_path, output_folder + '/oracle_acc_gan.png', 'Dataset', family='gan')
#             save_ap_auc_barchart(input_file_path, output_folder + '/ap_auc_gan.png', 'Dataset', family='gan')
#             save_metrics_barchart(input_file_path, output_folder + '/metrics_diffusion.png', 'Dataset', family='diffusion')
#             save_acc_barchart(input_file_path, output_folder + '/oracle_acc_diffusion.png', 'Dataset', family='diffusion')
#             save_ap_auc_barchart(input_file_path, output_folder + '/ap_auc_diffusion.png', 'Dataset', family='diffusion')
#             save_metrics_barchart(input_file_path, output_folder + '/metrics_synthbuster.png', 'Dataset', family='diffusion', source='synthbuster')
#             save_acc_barchart(input_file_path, output_folder + '/oracle_acc_synthbuster.png', 'Dataset', family='diffusion', source='synthbuster')
#             save_ap_auc_barchart(input_file_path, output_folder + '/ap_auc_synthbuster.png', 'Dataset', family='diffusion', source='synthbuster')
            

# avg_fl = './baseline_results/average_metrics.json'
# save_metrics_barchart(avg_fl, 'figs/average_metrics.png', 'Models', family='All', source='All')
# save_metrics_barchart(avg_fl, 'figs/average_metrics_gan.png', 'Models', family='gan', source='All')
# save_metrics_barchart(avg_fl, 'figs/average_metrics_diffusion.png', 'Models', family='diffusion', source='All')
# save_metrics_barchart(avg_fl, 'figs/average_metrics_synthbuster.png', 'Models', family='diffusion', source='synthbuster')

# save_acc_barchart(avg_fl, 'figs/average_oracle_acc.png', 'Models', family='All', source='All')
# save_acc_barchart(avg_fl, 'figs/average_oracle_acc_gan.png', 'Models', family='gan', source='All')
# save_acc_barchart(avg_fl, 'figs/average_oracle_acc_diffusion.png', 'Models', family='diffusion', source='All')
# save_acc_barchart(avg_fl, 'figs/average_oracle_acc_synthbuster.png', 'Models', family='diffusion', source='synthbuster')

# save_ap_auc_barchart(avg_fl, 'figs/average_ap_auc.png', 'Models', family='All', source='All')
# save_ap_auc_barchart(avg_fl, 'figs/average_ap_auc_gan.png', 'Models', family='gan', source='All')
# save_ap_auc_barchart(avg_fl, 'figs/average_ap_auc_diffusion.png', 'Models', family='diffusion', source='All')
# save_ap_auc_barchart(avg_fl, 'figs/average_ap_auc_synthbuster.png', 'Models', family='diffusion', source='synthbuster')


# avg_fl = './result/ojha2022_linear/average_metrics.json'
# save_metrics_barchart(avg_fl, 'figs/transformations/average_metrics.png', 'Transformation', family='All', source='All')
# save_metrics_barchart(avg_fl, 'figs/transformations/average_metrics_gan.png', 'Transformation', family='gan', source='All')
# save_metrics_barchart(avg_fl, 'figs/transformations/average_metrics_diffusion.png', 'Transformation', family='diffusion', source='All')
# save_metrics_barchart(avg_fl, 'figs/transformations/average_metrics_synthbuster.png', 'Transformation', family='diffusion', source='synthbuster')

# save_acc_barchart(avg_fl, 'figs/transformations/average_oracle_acc.png', 'Transformation', family='All', source='All')
# save_acc_barchart(avg_fl, 'figs/transformations/average_oracle_acc_gan.png', 'Transformation', family='gan', source='All')
# save_acc_barchart(avg_fl, 'figs/transformations/average_oracle_acc_diffusion.png', 'Transformation', family='diffusion', source='All')
# save_acc_barchart(avg_fl, 'figs/transformations/average_oracle_acc_synthbuster.png', 'Transformation', family='diffusion', source='synthbuster')

# save_ap_auc_barchart(avg_fl, 'figs/transformations/average_ap_auc.png', 'Transformation', family='All', source='All')
# save_ap_auc_barchart(avg_fl, 'figs/transformations/average_ap_auc_gan.png', 'Transformation', family='gan', source='All')
# save_ap_auc_barchart(avg_fl, 'figs/transformations/average_ap_auc_diffusion.png', 'Transformation', family='diffusion', source='All')
# save_ap_auc_barchart(avg_fl, 'figs/transformations/average_ap_auc_synthbuster.png', 'Transformation', family='diffusion', source='synthbuster')
