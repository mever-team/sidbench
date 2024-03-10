import argparse
import os
import json
from numpyencoder import NumpyEncoder

from dataset.dataset import SyntheticImagesDataset
from dataset.dataset_paths import DATASET_PATHS

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

def format_metric(metric, p=2):
    return str(round(metric*100, p))


def write_metrics(output_folder, all_metrics):
    if os.path.exists(output_folder):
       shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    with open( os.path.join(output_folder, 'ap.txt'), 'a') as f:
        for metrics in all_metrics:
            key = metrics['generative_model']
            f.write(key + ': ' + format_metric(metrics['ap']) + '\n' )

    with open( os.path.join(output_folder, 'acc_05.txt'), 'a') as f:
        for metrics in all_metrics:
            key = metrics['generative_model']
            m = metrics['threshold_05']
            f.write(key + ': ' + format_metric(m['r_acc'])  + '  ' + format_metric(m['f_acc']) + '  ' + format_metric(m['acc']) + '\n' )

    with open( os.path.join(output_folder, 'acc_oracle.txt'), 'a') as f:
        for metrics in all_metrics:   
            key = metrics['generative_model']
            m = metrics['oracle_threshold']
            f.write(key + ': ' + format_metric(m['r_acc'])  + '  ' + format_metric(m['f_acc']) + '  ' + format_metric(m['acc']) + '\n' )

    with open( os.path.join(output_folder, 'roc_auc.txt'), 'a') as f:
        for metrics in all_metrics:
            key = metrics['generative_model']
            f.write(key + ': ' + format_metric(metrics['roc_auc']) + '\n' )

    with open( os.path.join(output_folder, 'best_threshold.txt'), 'a') as f:
        for metrics in all_metrics:    
            key = metrics['generative_model']
            f.write(key + ': ' + str(round(metrics['best_threshold'], 3)) + '\n' )
    
    curves = [
        {
            'roc_curve': metrics.pop('roc_curve'), 
            'precision_recall_curve': metrics.pop('precision_recall_curve'),
            'generative_model': metrics['generative_model'],
            'source': metrics['source'],
            'family': metrics['family']
        } 
        for metrics in all_metrics
    ]

    with open(os.path.join(output_folder, 'metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=4, cls=NumpyEncoder)
    
    with open(os.path.join(output_folder, 'curves.json'), 'w') as f:
        json.dump(curves, f, indent=4, cls=NumpyEncoder)


def validate(model, loader, device, find_threshold=False):
    y_true, y_pred = [], []
    for img, label in tqdm(loader):
        img = img.to(device) 

        predictions = model.predict(img)
        y_pred.extend(predictions)

        y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return calculate_performance_metrics(y_true, y_pred, find_threshold)


def run_for_model(datasets, model, opt):
    
    device = setup_device(opt.gpus)

    all_metrics = []
    for dataset_params in datasets:
        set_random_seed()

        data_paths = dataset_params['data_paths']
        dataset = SyntheticImagesDataset(data_paths=data_paths, opt=opt, process_fn=processing)

        print('\t\t', dataset_params['source'], dataset_params['generative_model'], len(dataset))
        loader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=opt.batchSize, 
                                             shuffle=False, 
                                             num_workers=opt.numThreads)

        metrics = validate(model, loader, device, find_threshold=True)
        metrics['source'] = dataset_params['source'] if 'source' in dataset_params else 'unknown'
        metrics['generative_model'] = dataset_params['generative_model'] if 'generative_model' in dataset_params else 'unknown'
        metrics['family'] = dataset_params['family'] if 'family' in dataset_params else 'unknown'

        all_metrics.append(metrics)

    output_folder = os.path.join(opt.resultFolder, opt.modelName, '_jpeg_' + str(opt.jpegQuality) + '_gaussian_' + str(opt.gaussianSigma))
    
    write_metrics(output_folder, all_metrics)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = EvalOptions().initialize(parser)

    opt = parser.parse_args()

    print('model: ', opt.modelName)

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

