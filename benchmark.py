import argparse

from models import get_model
from models.models import MODELS

from dataset.dataset_paths import DATASET_PATHS

from evaluate import run_for_model

from options import TestOptions
from utils.util import set_random_seed

SEED = 0

JPEG_QUALITY = [95, 90, 50, 30]
GAUSSIAN_SIGMA = [2, 4]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = TestOptions().initialize(parser)

    opt = parser.parse_args()

    datasets = [
        dict(data_paths=[dp['real_path'], dp['fake_path']], 
             source=dp['source'],
             generative_model=dp['generative_model'],
             family=dp['family'])
        for dp in DATASET_PATHS
    ]

    for model_params in MODELS:
        set_random_seed()
        print('Model: ', model_params['modelName'])

        opt.modelName = model_params['modelName']
        opt.ckpt = model_params['ckpt']
        
        model = get_model(opt)

        print('\tjpeg_quality: ', None, 'gaussian_sigma: ', None)
        opt.gaussianSigma = None
        opt.jpegQuality = None
        run_for_model(datasets=datasets, model=model, opt=opt)

        for jpeg_quality in JPEG_QUALITY:
            print('\tjpeg_quality: ', jpeg_quality)
            opt.gaussianSigma = None
            opt.jpegQuality = jpeg_quality
            run_for_model(datasets=datasets, model=model, opt=opt)
        
        for gaussian_sigma in GAUSSIAN_SIGMA:
            print('\tgaussian_sigma: ', gaussian_sigma)
            opt.gaussianSigma = gaussian_sigma
            opt.jpegQuality = None
            run_for_model(datasets=datasets, model=model, opt=opt)
        
        print('\tjpeg_quality: ', 50, 'gaussian_sigma: ', 2)
        opt.gaussianSigma = 2
        opt.jpegQuality = 50
        run_for_model(datasets=datasets, model=model, opt=opt)