import argparse
import os
import json
from numpyencoder import NumpyEncoder

from dataset import SyntheticImageDetectionDataset
from datasets import LGRAD_DATASET_PATHS, DATASET_PATHS

import random
import shutil

import torch
from tqdm import tqdm

import numpy as np

from models import get_model
from eval_utils import calculate_performance_metrics


device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("device: ", device)

SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def format_metric(metric, p=2):
    return str(round(metric*100, p))


def write_metrics(output_folder, all_metrics):
    if os.path.exists(output_folder):
       shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    with open( os.path.join(output_folder, 'ap.txt'), 'a') as f:
        for metrics in all_metrics:
            key = metrics['key']
            f.write(key + ': ' + format_metric(metrics['ap']) + '\n' )

    with open( os.path.join(output_folder, 'acc_05.txt'), 'a') as f:
        for metrics in all_metrics:
            key = metrics['key']
            m = metrics['threshold_05']
            f.write(key + ': ' + format_metric(m['r_acc'])  + '  ' + format_metric(m['f_acc']) + '  ' + format_metric(m['acc']) + '\n' )

    with open( os.path.join(output_folder, 'acc_oracle.txt'), 'a') as f:
        for metrics in all_metrics:   
            key = metrics['key']
            m = metrics['oracle_threshold']
            f.write(key + ': ' + format_metric(m['r_acc'])  + '  ' + format_metric(m['f_acc']) + '  ' + format_metric(m['acc']) + '\n' )

    with open( os.path.join(output_folder, 'roc_auc.txt'), 'a') as f:
        for metrics in all_metrics:
            key = metrics['key']
            f.write(key + ': ' + format_metric(metrics['roc_auc']) + '\n' )

    with open( os.path.join(output_folder, 'best_threshold.txt'), 'a') as f:
        for metrics in all_metrics:    
            key = metrics['key']
            f.write(key + ': ' + str(round(metrics['best_threshold'], 3)) + '\n' )
    
    curves = [
        {
            'roc_curve': metrics.pop('roc_curve'), 
            'precision_recall_curve': metrics.pop('precision_recall_curve'),
            'key': metrics['key'],
            'source': metrics['source'],
            'family': metrics['family']
        } 
        for metrics in all_metrics
    ]

    with open(os.path.join(output_folder, 'metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=4, cls=NumpyEncoder)
    
    with open(os.path.join(output_folder, 'curves.json'), 'w') as f:
        json.dump(curves, f, indent=4, cls=NumpyEncoder)


def validate(model, loader, find_thres=False):
    y_true, y_pred = [], []
    for img, label in tqdm(loader):
        img = img.to(device) 

        predictions = model.predict(img)
        y_pred.extend(predictions)

        y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return calculate_performance_metrics(y_true, y_pred, find_thres)


def _run_for_model(dataset_paths, result_folder, model, model_name, max_sample, batch_size, jpeg_quality=None, gaussian_sigma=None):
    
    all_metrics = []
    for dataset_path in (dataset_paths):
        set_seed()
        dataset = SyntheticImageDetectionDataset(
            [dataset_path['real_path'], dataset_path['fake_path']], 
            max_sample, 
            is_train=False,
            jpeg_quality=jpeg_quality, 
            gaussian_sigma=gaussian_sigma
        )

        print('\t\t', dataset_path['source'], dataset_path['key'], len(dataset))

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        metrics = validate(model, loader, find_thres=True)
        metrics['source'] = dataset_path['source']
        metrics['key'] = dataset_path['key']
        metrics['family'] = dataset_path['family']

        all_metrics.append(metrics)

    output_folder = os.path.join(result_folder, model_name, '_jpeg_' + str(jpeg_quality) + '_gaussian_' + str(gaussian_sigma))
    
    write_metrics(output_folder, all_metrics)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--real_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--fake_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--source', type=str, default='synthbuster', help='wang2020, ojha2023 or synthbuster')

    parser.add_argument('--key', type=str, default=None)
    parser.add_argument('--max_sample', type=int, default=None, help='only check this number of images for both fake/real')
    
    parser.add_argument('--ckpt', type=str, default='./weights/ojha2022/fc_weights.pth')
    parser.add_argument('--model_name', type=str, default='ojha2022')

    parser.add_argument('--result_folder', type=str, default='test_results', help='')
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--jpeg_quality', type=int, default=None, help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None, help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")

    opt = parser.parse_args()

    print('model: ', opt.model)

    model = get_model(model_name=opt.model_name, ckpt=opt.ckpt)
    model = model.to(device)

    if (opt.real_path == None) or (opt.fake_path == None):
        dataset_paths = [dp for dp in DATASET_PATHS if dp['source'] == opt.source] if opt.source != None else DATASET_PATHS
    else:
        dataset_paths = [ dict(real_path=opt.real_path, fake_path=opt.fake_path, source=opt.source, key=opt.key) ]

    _run_for_model(dataset_paths=dataset_paths, result_folder=opt.result_folder,
                   model=model, model_name=opt.model_name, classifier=opt.classifier, backbone=opt.backbone, 
                   max_sample=opt.max_sample, batch_size=opt.batch_size, 
                   jpeg_quality=opt.jpeg_quality, gaussian_sigma=opt.gaussian_sigma)

