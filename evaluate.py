import argparse
import os
import json
from numpyencoder import NumpyEncoder

from dataset.dataset import SyntheticImagesDataset
from dataset.dataset_paths import DATASET_PATHS
from dataset import patch_collate

import shutil

import torch
from tqdm import tqdm

import numpy as np

from models import get_model
from utils.evaluation_utils import calculate_performance_metrics

from options import EvalOptions
from utils.util import set_random_seed, setup_device

from dataset.process import processing

SEED = 0

def write_metrics(output_folder, all_metrics):
    if os.path.exists(output_folder):
       shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    with open( os.path.join(output_folder, 'ap.txt'), 'a') as f:
        headers = [['Generative', 'Average'], ['Model', 'Precision']]
        column_widths = [12, 12] 

        for header_names in headers:
            header_line = f"{header_names[0]:<{column_widths[0]}} {header_names[1]:>{column_widths[1]}}"
            f.write(header_line + '\n')
        f.write('-' * sum(column_widths) + '\n')
        for metrics in all_metrics:
            key = metrics['generative_model'] if 'generative_model' in metrics and metrics['generative_model'] is not None else 'unknown'
            ap_formatted = f"{metrics['ap']*100:6.2f}"
            f.write(f"{key:<{12}} {ap_formatted:>{12}}" + '\n')

    # Write accuracies to a file
    with open( os.path.join(output_folder, 'acc_05.txt'), 'a') as f:
        headers = [['Generative', 'TNR', 'TPR', ''], ['Model', '(Reals)', 'Fakes', 'Accuracy']]
        column_widths = [12, 12, 12, 12] 

        for header_names in headers:
            header_line = f"{header_names[0]:<{column_widths[0]}} {header_names[1]:>{column_widths[1]}} {header_names[2]:>{column_widths[2]}} {header_names[3]:>{column_widths[3]}}"
            f.write(header_line + '\n')
        f.write('-' * sum(column_widths) + '\n')
        for metrics in all_metrics:
            key = metrics['generative_model'] if 'generative_model' in metrics and metrics['generative_model'] is not None else 'unknown'
            m = metrics['threshold_05']

            # Formatting each metric with a specified precision (e.g., 2 decimal places)
            r_acc_formatted = f"{m['r_acc']*100:6.2f}"  # Multiplies by 100 to convert to percentage
            f_acc_formatted = f"{m['f_acc']*100:6.2f}"
            acc_formatted = f"{m['acc']*100:6.2f}"
            
            data_line = f"{key:<{column_widths[0]}} {r_acc_formatted:>{column_widths[1]}} {f_acc_formatted:>{column_widths[2]}} {acc_formatted:>{column_widths[3]}}"
            f.write(data_line + '\n')

    with open( os.path.join(output_folder, 'acc_oracle.txt'), 'a') as f:
        headers = [['Generative', 'TNR', 'TPR', '', 'Best'], ['Model', '(Reals)', 'Fakes', 'Accuracy', 'Threshold']]
        column_widths = [12, 12, 12, 12, 12] 

        for header_names in headers:
            header_line = f"{header_names[0]:<{column_widths[0]}} {header_names[1]:>{column_widths[1]}} {header_names[2]:>{column_widths[2]}} {header_names[3]:>{column_widths[3]}} {header_names[4]:>{column_widths[4]}}"
            f.write(header_line + '\n')
        f.write('-' * sum(column_widths) + '\n')
        for metrics in all_metrics:   
            key = metrics['generative_model'] if 'generative_model' in metrics and metrics['generative_model'] is not None else 'unknown'
            m = metrics['oracle_threshold']
            
            # Formatting each metric with a specified precision (e.g., 2 decimal places)
            r_acc_formatted = f"{m['r_acc']*100:6.2f}"  # Multiplies by 100 to convert to percentage
            f_acc_formatted = f"{m['f_acc']*100:6.2f}"
            acc_formatted = f"{m['acc']*100:6.2f}"
            best_threshold = f"{metrics['best_threshold']:6.3f}"

            data_line = f"{key:<{column_widths[0]}} {r_acc_formatted:>{column_widths[1]}} {f_acc_formatted:>{column_widths[2]}} {acc_formatted:>{column_widths[3]}} {best_threshold:>{column_widths[4]}}"
            f.write(data_line + '\n')
    
    curves = [
        {
            'roc_curve': metrics.pop('roc_curve'), 
            'precision_recall_curve': metrics.pop('precision_recall_curve'),
            'generative_model': metrics['generative_model'] if 'generative_model' in metrics and metrics['generative_model'] is not None else 'unknown',
            'source': metrics['source'] if 'source' in metrics else 'unknown',
            'family': metrics['family'] if 'family' in metrics else 'unknown'
        } 
        for metrics in all_metrics
    ]

    with open(os.path.join(output_folder, 'metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=4, cls=NumpyEncoder)
    
    with open(os.path.join(output_folder, 'curves.json'), 'w') as f:
        json.dump(curves, f, indent=4, cls=NumpyEncoder)


def validate(model, loader, device, dataset_length, find_threshold=False):
    y_true, y_pred = [], []
    with tqdm(total=dataset_length) as pbar:
        for img, label, _ in loader:
            # if list move each part to device
            if isinstance(img, list):
                img = [
                    # i.to(device) if isinstance(i, torch.Tensor) else [j.to(device) for j in i]
                    i.to(device) if isinstance(i, torch.Tensor) else i for i in img
                ]
                predictions = model.predict(*img)
            else:
                img = img.to(device) 
                predictions = model.predict(img)    

            y_pred.extend(predictions)
            y_true.extend(label.flatten().tolist())
            
            pbar.update(len(predictions))

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return calculate_performance_metrics(y_true, y_pred, find_threshold)


def get_results_path(opt):
    components = []

    if opt.cropSize is not None:
        components.append(f"crop_{opt.cropSize}")
    else:
        components.append("noCrop")

    if opt.loadSize is not None:
        components.append(f"resize_{opt.loadSize}")
    else:
        components.append("noResize")

    if opt.jpegQuality is not None:
        components.append(f"jpeg_{opt.jpegQuality}")

    if opt.gaussianSigma is not None:
        components.append(f"gaussian_{opt.gaussianSigma}")

    output_folder = os.path.join(opt.resultFolder, opt.modelName, '_'.join(components))
    return output_folder



def run_for_model(datasets, model, opt):
    device = setup_device(opt.gpus)

    collate_fn = patch_collate if opt.modelName == 'Fusing' else None

    all_metrics = []
    for dataset_params in datasets:
        set_random_seed()

        data_paths = dataset_params['data_paths']
        dataset = SyntheticImagesDataset(data_paths=data_paths, opt=opt, process_fn=processing)

        source = dataset_params.get('source', 'N/A') or 'N/A'
        generative_model = dataset_params.get('generative_model', 'N/A') or 'N/A'
        dataset_length = len(dataset) if dataset is not None else 'N/A'
        print(f'Source: {source:<20} Generative Model: {generative_model:<20} Dataset Length: {dataset_length:<15}')

        loader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=opt.batchSize, 
                                             shuffle=False, 
                                             num_workers=opt.numThreads,
                                             collate_fn=collate_fn)

        metrics = validate(model, loader, device, dataset_length=dataset_length, find_threshold=True)
        metrics['source'] = dataset_params['source'] if 'source' in dataset_params else 'unknown'
        metrics['generative_model'] = dataset_params['generative_model'] if 'generative_model' in dataset_params else 'unknown'
        metrics['family'] = dataset_params['family'] if 'family' in dataset_params else 'unknown'

        all_metrics.append(metrics)

    output_folder = get_results_path(opt)
    write_metrics(output_folder, all_metrics)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = EvalOptions().initialize(parser)

    opt = parser.parse_args()

    model = get_model(opt)

    if (opt.realPath != None) and (opt.fakePath != None):
        datasets = [ 
            dict(data_paths=[opt.realPath, opt.fakePath], 
                 source=opt.source,
                 generative_model=opt.generativeModel,
                 family=opt.family
            )
        ]
    elif opt.dataPath != None:
        datasets = [ 
            dict(data_paths=[opt.dataPath], 
                 source=opt.source,
                 generative_model=opt.generativeModel,
                 family=opt.family
            )
        ]
    else:
        dataset_paths = [dp for dp in DATASET_PATHS if dp['source'] == opt.source] if opt.source != None else DATASET_PATHS
        datasets = [
            dict(data_paths=[dp['real_path'], dp['fake_path']], 
                 source=dp['source'],
                 generative_model=dp['generative_model'],
                 family=dp['family'])
            for dp in dataset_paths
        ]
    
    run_for_model(datasets=datasets, model=model, opt=opt)

