import argparse
import os
import json
from numpyencoder import NumpyEncoder

from dataset import SyntheticImageDetectionDataset
from datasets import DATASET_PATHS

import random
import shutil

import torch

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


def write_metrics(output_folder, key, metrics):
    with open( os.path.join(output_folder, 'ap.txt'), 'a') as f:
        f.write(key + ': ' + format_metric(metrics['ap']) + '\n' )

    with open( os.path.join(output_folder, 'acc_05.txt'), 'a') as f:
        m = metrics['threshold_05']
        f.write(key + ': ' + format_metric(m['r_acc'])  + '  ' + format_metric(m['f_acc']) + '  ' + format_metric(m['acc']) + '\n' )

    with open( os.path.join(output_folder, 'acc_oracle.txt'), 'a') as f:
        m = metrics['oracle_threshold']
        f.write(key + ': ' + format_metric(m['r_acc'])  + '  ' + format_metric(m['f_acc']) + '  ' + format_metric(m['acc']) + '\n' )

    with open( os.path.join(output_folder, 'roc_auc.txt'), 'a') as f:
        f.write(key + ': ' + format_metric(metrics['roc_auc']) + '\n' )

    with open( os.path.join(output_folder, 'best_threshold.txt'), 'a') as f:
        f.write(key + ': ' + str(round(metrics['best_threshold'], 3)) + '\n' )


def validate(model, loader, find_thres=False):
    y_true, y_pred = [], []
    print ("Batches: %d" %(len(loader)))

    for img, label in loader:
        img = img.to(device) 

        predictions = model.predict(img)
        y_pred.extend(predictions)

        y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return calculate_performance_metrics(y_true, y_pred, find_thres)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--real_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--fake_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--source', type=str, default=None, help='wang2020, ojha2023 or synthbuster')

    parser.add_argument('--key', type=str, default=None)
    parser.add_argument('--max_sample', type=int, default=1000, help='only check this number of images for both fake/real')

    # parser.add_argument('--backbone', type=str, default='CLIP:ViT-L/14')
    # parser.add_argument('--ckpt', type=str, default='./pretrained_weights/ojha2022/fc_weights.pth')
    # parser.add_argument('--classifier', type=str, default='linear')
    # parser.add_argument('--model_name', type=str, default='ojha2022')

    parser.add_argument('--backbone', type=str, default='Imagenet:resnet50')
    parser.add_argument('--ckpt', type=str, default='./pretrained_weights/wang2020/blur_jpg_prob0.1.pth')
    parser.add_argument('--classifier', type=str, default='linear')
    parser.add_argument('--model_name', type=str, default='wang2020_jpg_prob01')

    # parser.add_argument('--backbone', type=str, default='Imagenet:resnet50nodown')
    # parser.add_argument('--ckpt', type=str, default='./pretrained_weights/corvi2022/latent_model_epoch_best.pth')
    # parser.add_argument('--classifier', type=str, default='linear')
    # parser.add_argument('--model_name', type=str, default='corvi2022')

    # parser.add_argument('--backbone', type=str, default='Imagenet:resnet50nodown')
    # parser.add_argument('--ckpt', type=str, default='./pretrained_weights/grag2021/gandetection_resnet50nodown_stylegan2.pth')
    # parser.add_argument('--classifier', type=str, default='linear')
    # parser.add_argument('--model_name', type=str, default='grag2021_stylegan2')

    parser.add_argument('--result_folder', type=str, default='result', help='')
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--jpeg_quality', type=int, default=None, help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None, help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")

    opt = parser.parse_args()

    print('classifier: ', opt.classifier)
    print('backbone: ', opt.backbone)
    print('model_name: ', opt.model_name)

    model = get_model(classifier=opt.classifier, backbone=opt.backbone, ckpt=opt.ckpt)
    model = model.to(device)

    output_folder = os.path.join(opt.result_folder, opt.model_name + '_' + opt.classifier)

    if os.path.exists(output_folder):
       shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    if (opt.real_path == None) or (opt.fake_path == None) or (opt.source == None):
        dataset_paths = DATASET_PATHS
    else:
        dataset_paths = [ dict(real_path=opt.real_path, fake_path=opt.fake_path, source=opt.source, key=opt.key) ]

    all_metrics = []

    for dataset_path in (dataset_paths):

        set_seed()

        print(dataset_path['source'], dataset_path['key'])  
        dataset = SyntheticImageDetectionDataset(
            [dataset_path['real_path'], dataset_path['fake_path']], 
            opt.backbone,
            opt.max_sample, 
            is_train=False,
            jpeg_quality=opt.jpeg_quality, 
            gaussian_sigma=opt.gaussian_sigma,
        )
        print('Length of dataset: ', len(dataset))

        loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

        metrics = validate(model, loader, find_thres=True)

        metrics['source'] = dataset_path['source']
        metrics['key'] = dataset_path['key']

        all_metrics.append(metrics)

        write_metrics(output_folder, dataset_path['key'], metrics)
    
    with open( os.path.join(output_folder, 'metrics.json'), 'a') as f:
        json.dump(all_metrics, f, indent=4, cls=NumpyEncoder)


