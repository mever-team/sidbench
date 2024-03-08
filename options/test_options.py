class TestOptions():
    
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--realPath', type=str, default=None, help='dir name of reals')
        parser.add_argument('--fakePath', type=str, default=None, help='dir name of fakes')
        parser.add_argument('--source',  default='ours', help='wang2020 or ours')
        parser.add_argument('--maxSample', type=int, default=None, help='only check this number of images for both fake/real')

        parser.add_argument('--modelName', type=str, default='UnivFD')    
        parser.add_argument('--ckpt', type=str, default='./weights/ojha2022/fc_weights.pth')

        parser.add_argument('--resultFolder', type=str, default='test_results', help='')
        
        parser.add_argument('--batchSize', type=int, default=32)

        parser.add_argument('--jpegQuality', type=int, default=None, help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
        parser.add_argument('--gaussianSigma', type=int, default=None, help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")

        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--cropSize', type=int, default=224, help='then crop to this size')
        parser.add_argument('--noResize', action='store_true')
        parser.add_argument('--noCrop', action='store_true')

        parser.add_argument('--gpu', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--numThreads', default=4, type=int, help='# threads for loading data')

        self.isTrain = False
        return parser
