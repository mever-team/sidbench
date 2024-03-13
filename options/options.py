import argparse
import os
import utils.util as util
import torch


class TestOptions():
    
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):

        parser.add_argument('--dataPath', type=str, default=None, help='dir name of data')
        parser.add_argument('--modelName', type=str, default='UnivFD', help='One of CNNDetect, FreqDetect, Fusing, GramNet, LGrad, UnivFD, RPTC, Rine, DIMD, NPR, Dire')    
        parser.add_argument('--ckpt', type=str, default='./weights/univfd/fc_weights.pth', help='Must match with the selected model')

        parser.add_argument('--predictionsFile', type=str, default='predictions.csv', help='file to save predictions')

        parser.add_argument('--loadSize', type=int, default=None, help='scale images to this size')
        parser.add_argument('--cropSize', type=int, default=224, help='crop to this size')

        parser.add_argument('--gpus', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--numThreads', default=1, type=int, help='# threads for loading data')
        parser.add_argument('--batchSize', type=int, default=64)

        parser.add_argument('--isTrain', default=False, type=bool, help='train or test')
        
        # additional parameters

        # RPTC
        parser.add_argument('--patchNum', type=int, default=3)
        
        # FreqDetect
        parser.add_argument('--dctMean', type=str, default='./weights/freqdetect/dct_mean')
        parser.add_argument('--dctVar', type=str, default='./weights/freqdetect/dct_var')
        
        # LGrad
        parser.add_argument('--LGradGenerativeModelPath', type=str,default='./weights/preprocessing/karras2019stylegan-bedrooms-256x256_discriminator.pth', help='the path of LGrad pre-trained model')
        parser.add_argument('--LGradGenerativeModel')

        # Dire
        parser.add_argument('--DireGenerativeModelPath', type=str, default='./weights/preprocessing/lsun_bedroom.pt')
        parser.add_argument('--diffusion')
        parser.add_argument('--diffusionModel')
        parser.add_argument('--direArgs')

        # DeFake
        parser.add_argument('--defakeClipEncoderPath', type=str,default='./weights/defake/finetune_clip.pt', help='the path of defake clip encoder model')
        parser.add_argument('--defakeClipEncode')
        parser.add_argument('--defakeBlipPath', type=str,default='./weights/defake/model_base_capfilt_large.pth', help='the path of defake blip model')
        parser.add_argument('--defakeBlip')

        self.initialized = True
        
        return parser
        

class EvalOptions():
    
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--realPath', type=str, default=None, help='dir name of reals')
        parser.add_argument('--fakePath', type=str, default=None, help='dir name of fakes')
        parser.add_argument('--dataPath', type=str, default=None, help='dir name of all data')
        parser.add_argument('--source',  default=None, help='wang2020, ojha2023, synthbuster')
        parser.add_argument('--generativeModel',  default=None)
        parser.add_argument('--family',  default=None, help='family of generative model: gan, deepfake, perceptual_loss, low_level_vision, diffusion')
        parser.add_argument('--maxSample', type=int, default=None, help='only check this number of images for both fake/real')

        parser.add_argument('--modelName', type=str, default='UnivFD', help='One of CNNDetect, FreDetect, Fusing, GramNet, LGrad, UnivFD, RPTC, Rine, DIMD, NPR')    
        parser.add_argument('--ckpt', type=str, default='./weights/univfd/fc_weights.pth', help='Must match with the selected model')

        parser.add_argument('--resultFolder', type=str, default='test_results', help='')
        
        parser.add_argument('--batchSize', type=int, default=64)

        parser.add_argument('--jpegQuality', type=int, default=None, help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
        parser.add_argument('--gaussianSigma', type=int, default=None, help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")

        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--cropSize', type=int, default=224, help='crop images to this size')
        parser.add_argument('--noResize', default=False, action='store_true')
        parser.add_argument('--noCrop', default=False, action='store_true')

        parser.add_argument('--gpus', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--numThreads', default=4, type=int, help='# threads for loading data')

        parser.add_argument('--isTrain', default=False, type=bool, help='train or test')

        # additional parameters

        # RPTC
        parser.add_argument('--patchNum', type=int, default=3)
        
        # FreqDetect
        parser.add_argument('--dctMean', type=str, default='./weights/freqdetect/dct_mean')
        parser.add_argument('--dctVar', type=str, default='./weights/freqdetect/dct_var')
        
        # LGrad
        parser.add_argument('--LGradGenerativeModelPath', type=str, default='./weights/preprocessing/karras2019stylegan-bedrooms-256x256_discriminator.pth', help='the path of LGrad pre-trained model')
        parser.add_argument('--LGradGenerativeModel')

        # Dire
        parser.add_argument('--DireGenerativeModelPath', type=str, default='./weights/preprocessing/lsun_bedroom.pt')
        parser.add_argument('--diffusion')
        parser.add_argument('--diffusionModel')
        parser.add_argument('--direArgs')

        # DeFake
        parser.add_argument('--defakeClipEncoderPath', type=str,default='./weights/defake/finetune_clip.pt', help='the path of defake clip encoder model')
        parser.add_argument('--defakeClipEncode')
        parser.add_argument('--defakeBlipPath', type=str,default='./weights/defake/model_base_capfilt_large.pth', help='the path of defake blip model')
        parser.add_argument('--defakeBlip')

        self.initialized = True

        return parser


class TrainOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):

        # data augmentation
        parser.add_argument('--rzInterp', default='bilinear')
        parser.add_argument('--blurProb', type=float, default=0.5)
        parser.add_argument('--blurSig', default='0.0,3.0')
        parser.add_argument('--jpgProb', type=float, default=0.5)
        parser.add_argument('--jpgMethod', default='cv2,pil')
        parser.add_argument('--jpgQual', default='30,100')
        
        parser.add_argument('--realPath', default=None, help='only used if data_mode==ours: path for the list of real images, which should contain train.pickle and val.pickle')
        parser.add_argument('--fakePath', default=None, help='only used if data_mode==ours: path for the list of fake images, which should contain train.pickle and val.pickle')
        parser.add_argument('--source',  default='ours', help='wang2020 or ours')
        parser.add_argument('--dataLabel', default='train', help='label to decide whether train or validation dataset')
        parser.add_argument('--weightDecay', type=float, default=0.0, help='loss weight for l2 reg')
        
        parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--cropSize', type=int, default=224, help='then crop to this size')
        parser.add_argument('--noFlip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--gpus', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--numThreads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='models are saved here')

        parser.add_argument('--serialBatches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--initType', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--initGain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
        
        parser.add_argument('--earlystopEpoch', type=int, default=5)
        parser.add_argument('--dataAug', action='store_true', help='if specified, perform additional data augmentation (photometric, blurring, jpegging)')
        parser.add_argument('--optim', type=str, default='adam', help='optim to use [sgd, adam]')
        parser.add_argument('--lossFreq', type=int, default=400, help='frequency of showing loss on tensorboard')
        parser.add_argument('--saveEpochFreq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--epochCount', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--last_epoch', type=int, default=-1, help='starting epoch count for scheduler intialization')
        parser.add_argument('--trainSplit', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--valSplit', type=str, default='val', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=100, help='total epoches')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')

        parser.add_argument('--isTrain', default=True, type=bool, help='train or test')

        self.initialized = True

        return parser


    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()


    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        if print_options:
            self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # additional
        #opt.classes = opt.classes.split(',')
        opt.rz_interp = opt.rz_interp.split(',')
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(',')]
        opt.jpg_method = opt.jpg_method.split(',')
        opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(',')]
        if len(opt.jpg_qual) == 2:
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        self.opt = opt
        return self.opt
