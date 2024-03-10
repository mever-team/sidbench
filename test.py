import argparse

from dataset.dataset import RecursiveImageDataset
from dataset.process import processing

from models import get_model
from options import TestOptions
import torch
from tqdm import tqdm

from utils.util import setup_device
from dataset import patch_collate


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = TestOptions().initialize(parser)

    opt = parser.parse_args()

    print('model: ', opt.modelName)
    print('ckpt: ', opt.ckpt)
    print('dataPath: ', opt.dataPath)

    device = setup_device(opt.gpus)
    
    model = get_model(opt)

    collate_fn = patch_collate if opt.modelName == 'Fusing' else None

    dataset = RecursiveImageDataset(data_path=opt.dataPath, opt=opt, process_fn=processing)
    loader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=opt.batchSize, 
                                         shuffle=False, 
                                         num_workers=opt.numThreads,
                                         collate_fn=collate_fn)
    
    all = 0
    correct = 0

    for img, label, img_path in tqdm(loader):
        # if list move each part to device
        if isinstance(img, list):
            img = [i.to(device) for i in img]
            predictions = model.predict(*img)
        else:
            img = img.to(device) 
            predictions = model.predict(img)
        

        labels = [1 if p > 0.5 else 0 for p in predictions]

        correct += sum(labels)
        all += len(labels)

    print('Accuracy: ', correct / all)