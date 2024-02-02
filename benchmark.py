import argparse

from models import get_model, MODELS

from datasets import DATASET_PATHS

import torch
import numpy as np
import random

from evaluate import _run_for_model


device = "cuda:0" if torch.cuda.is_available() else "cpu"

SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


JPEG_QUALITY = [90, 50, 30]
GAUSSIAN_SIGMA = [2, 4]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--max_sample', type=int, default=5, help='only check this number of images for both fake/real')
    parser.add_argument('--result_folder', type=str, default='result', help='')
    parser.add_argument('--batch_size', type=int, default=32)

    opt = parser.parse_args()

    for model_params in MODELS:
        set_seed()
        print('=' * 20)
        print('MODEL: ', model_params['name'])
        model = get_model(classifier=model_params['classifier'], 
                          backbone=model_params['backbone'], 
                          ckpt=model_params['ckpt'])
        model = model.to(device)

        for jpeg_quality in JPEG_QUALITY:
            print('\tjpeg_quality: ', jpeg_quality)
            _run_for_model(dataset_paths=DATASET_PATHS,
                           result_folder=opt.result_folder,
                           model=model, 
                           model_name=model_params['name'],
                           classifier=model_params['classifier'],
                           backbone=model_params['backbone'], 
                           max_sample=opt.max_sample, 
                           batch_size=opt.batch_size, 
                           jpeg_quality=jpeg_quality)
        
        for gaussian_sigma in GAUSSIAN_SIGMA:
            print('\tgaussian_sigma: ', gaussian_sigma)
            _run_for_model(dataset_paths=DATASET_PATHS,
                           result_folder=opt.result_folder,
                           model=model, 
                           model_name=model_params['name'],
                           classifier=model_params['classifier'],
                           backbone=model_params['backbone'], 
                           max_sample=opt.max_sample, 
                           batch_size=opt.batch_size, 
                           gaussian_sigma=gaussian_sigma)
        
        print('\tjpeg_quality: ', 50, 'gaussian_sigma: ', 2)
        _run_for_model(dataset_paths=DATASET_PATHS, result_folder=opt.result_folder,
                       model=model, model_name=model_params['name'], classifier=model_params['classifier'],
                       backbone=model_params['backbone'], max_sample=opt.max_sample, batch_size=opt.batch_size, 
                       jpeg_quality=50, gaussian_sigma=2)
        