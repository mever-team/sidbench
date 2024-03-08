import argparse

from models import get_model, MODELS
from datasets import DATASET_PATHS

import torch
from evaluate import run_for_model

from options import TestOptions
from util import set_random_seed


device = "cuda:0" if torch.cuda.is_available() else "cpu"

SEED = 0

JPEG_QUALITY = [90, 50, 30]
GAUSSIAN_SIGMA = [2, 4]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = TestOptions().initialize(parser)

    opt = parser.parse_args()

    for model_params in MODELS:
        set_random_seed()
        print('Model: ', model_params['model_name'])
        del model_params["trained_on"]

        model = get_model(**model_params)
        model = model.to(device)

        for jpeg_quality in JPEG_QUALITY:
            print('\tjpeg_quality: ', jpeg_quality)
            opt.gaussianSigma = None
            opt.jpegQuality = jpeg_quality
            run_for_model(dataset_paths=DATASET_PATHS, model=model, opt=opt)
        
        for gaussian_sigma in GAUSSIAN_SIGMA:
            print('\tgaussian_sigma: ', gaussian_sigma)
            opt.gaussianSigma = gaussian_sigma
            opt.jpegQuality = None
            run_for_model(dataset_paths=DATASET_PATHS, model=model, opt=opt)
        
        print('\tjpeg_quality: ', 50, 'gaussian_sigma: ', 2)
        opt.gaussianSigma = 2
        opt.jpegQuality = 50
        run_for_model(dataset_paths=DATASET_PATHS, model=model, opt=opt)
        
        print('\tjpeg_quality: ', None, 'gaussian_sigma: ', None)
        opt.gaussianSigma = None
        opt.jpegQuality = None
        run_for_model(dataset_paths=DATASET_PATHS, model=model, opt=opt)