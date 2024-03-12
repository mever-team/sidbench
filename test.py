import argparse

from dataset.dataset import RecursiveImageDataset
from dataset.process import processing
from dataset import patch_collate

from models import get_model
from options import TestOptions
import torch
from tqdm import tqdm

from utils.util import setup_device
import csv


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

    collected_predictions = []
    with tqdm(total=len(dataset)) as pbar:
        for img, label, img_path in loader:
            # if list move each part to device
            if isinstance(img, list):
                img = [i.to(device) if isinstance(i, torch.Tensor) else i for i in img]
                predictions = model.predict(*img)
            else:
                img = img.to(device) 
                predictions = model.predict(img)        

            labels = [1 if p > 0.5 else 0 for p in predictions]

            for path, prediction, label in zip(img_path, predictions, labels):
                collected_predictions.append((path, prediction, label))
            
            pbar.update(len(labels))

    # write the collected data to a CSV file
    with open(opt.predictionsFile, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Path", "Prediction", "Label"]) 
        for data in collected_predictions:
            writer.writerow(data)