import argparse

from dataset.dataset import RecursiveImageDataset
from dataset.process import processing

from models import get_model
from options import TestOptions
import torch
from tqdm import tqdm

from utils.util import setup_device
from dataset import patch_collate

from preprocessing.lgrad.models import build_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = TestOptions().initialize(parser)

    opt = parser.parse_args()

    device = setup_device(opt.gpus)

    print('model: ', opt.modelName)
    print('ckpt: ', opt.ckpt)
    print('dataPath: ', opt.dataPath)

    model = get_model(model_name=opt.modelName, ckpt=opt.ckpt)
    model = model.to(device)

    collate_fn = patch_collate if opt.modelName == 'Fusing' else None

    if opt.modelName == 'FreqDetect':
        opt.dctMean = torch.load(opt.dctMean).permute(1,2,0).numpy()
        opt.dctVar = torch.load(opt.dctVar).permute(1,2,0).numpy()

    if opt.modelName == 'LGrad':
        opt.numThreads = int(0)
        gen_model = build_model(gan_type='stylegan', 
                                module='discriminator', 
                                resolution=256, 
                                label_size=0, 
                                image_channels=3)
        gen_model.load_state_dict(torch.load(opt.LGradModelPath), strict=True)
        gen_model = gen_model.to(device)
        opt.LGradGenerativeModel = gen_model


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