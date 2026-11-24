import argparse
import copy
import os

from models import get_model, VALID_MODELS
from models.models import MODELS

from evaluate import run_for_model
from dataset.sources import build_datasets, DATASET_FIELDS, DATA_OPTIONS

from options import EvalOptions
from options.config import parse_with_config
from utils.util import set_random_seed

SEED = 0

JPEG_QUALITY = [95, 90, 50, 30]
GAUSSIAN_SIGMA = [2, 4]

# Clean images, each JPEG quality, each blur level, then JPEG quality 50 combined with blur sigma 2
DEGRADATIONS = (
    [dict()]
    + [dict(jpegQuality=jpeg_quality) for jpeg_quality in JPEG_QUALITY]
    + [dict(gaussianSigma=gaussian_sigma) for gaussian_sigma in GAUSSIAN_SIGMA]
    + [dict(jpegQuality=50, gaussianSigma=2)]
)

# Fields of each entry in the 'models' and 'degradations' lists of a config file: (types, required fields)
MODEL_FIELDS = (dict(modelName=str, ckpt=str, trainedOn=str), {'modelName', 'ckpt', 'trainedOn'})
DEGRADATION_FIELDS = (dict(jpegQuality=int, gaussianSigma=int), set())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = EvalOptions().initialize(parser)

    opt, config_lists = parse_with_config(
        parser, lists={'datasets': DATASET_FIELDS, 'models': MODEL_FIELDS, 'degradations': DEGRADATION_FIELDS},
        replace_groups=[DATA_OPTIONS]
    )

    if opt.jpegQuality is not None or opt.gaussianSigma is not None:
        parser.error("benchmark.py sets jpegQuality and gaussianSigma itself; use the 'degradations' list in a config file instead")

    models = config_lists.get('models', MODELS)
    degradations = config_lists.get('degradations', DEGRADATIONS)

    seen = set()
    for model_params in models:
        if model_params['modelName'] not in VALID_MODELS:
            parser.error(f"unknown modelName '{model_params['modelName']}'. Valid models: {', '.join(VALID_MODELS)}")
        key = (model_params['modelName'], model_params['trainedOn'])
        if key in seen:
            parser.error(f'two models have modelName={key[0]} and trainedOn={key[1]}; their results would overwrite each other')
        seen.add(key)

    datasets = build_datasets(opt, config_lists.get('datasets'))

    for model_params in models:
        set_random_seed()

        # get_model() changes some options (e.g. LGrad sets cropSize=256), so each model gets its own copy
        model_opt = copy.copy(opt)
        model_opt.modelName = model_params['modelName']
        model_opt.ckpt = model_params['ckpt']
        # One folder per training set, so checkpoints of the same model don't overwrite each other's results
        model_opt.resultFolder = os.path.join(opt.resultFolder, model_params['trainedOn'])

        model = get_model(model_opt)

        for degradation in degradations:
            model_opt.jpegQuality = degradation.get('jpegQuality')
            model_opt.gaussianSigma = degradation.get('gaussianSigma')
            print('\tjpeg_quality: ', model_opt.jpegQuality, 'gaussian_sigma: ', model_opt.gaussianSigma)
            run_for_model(datasets=datasets, model=model, opt=model_opt)
