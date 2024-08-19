import argparse


parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='VDSR',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=24,
                    help='number of threads for data loading')

# The number of threads required for data loading refers to the parallel threads that are used to load data efficiently
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')

# When the args command you mentioned is added after the instruction, it typically indicates the CPU to be used for training or other operations.
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
# The num of gpu for used
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
# set the num of random seed

# Data specifications
parser.add_argument('--dir_data', type=str, default='../dataset/',
                    help='dataset directory')

# The folder used to store the training dataset
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')

# The folder used to store the test dataset
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')

# The name of train dataset
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')

# The name of test dataset
parser.add_argument('--data_range', type=str, default='1-800/801-900',
                    help='train/test data range')

# The DIV2K dataset consists of 800 high-resolution images in the training subset,
# which are used for training super-resolution models. Additionally, there is a validation subset called DIV2K valid,
# which contains 100 high-resolution images. This validation subset is used to assess and evaluate the performance of models.

parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')

#  specify the filenames to be used for augmenting the dataset
parser.add_argument('--scale', type=str, default='2',
                    help='super resolution scale')

parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=256,
                    help='maximum value of RGB')
parser.add_argument('--ori_rgb_range', type=int, default=256,
                    help='cal psnr')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')

# rgb gray
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')

# start memory-efficient forward to make reduce memory consumption and improve computational efficiency
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Model specifications
parser.add_argument('--model', default='EDSR',
                    help='model name')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')

# define activation function
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')

# The storage path for a trained model
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')

parser.add_argument('--n_resblocks', type=int, default=30,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=290,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=0.3,
                    help='residual scaling')

parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')



parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')
# RCNN param
parser.add_argument('--RCNN_channel', type=str, default="on",
                    help='add RCNN channel into the training dataset.')

parser.add_argument('--resize',  type=str, default="off",
                    help='resize all images into same shape')
parser.add_argument('--beta', type=float, default=0.1,
                    help='Weighting factor that controls the relationship between feedback and link inputs')
parser.add_argument('--alpha_theta', type=float, default=0.04,
                    help='Dynamic threshold decay coefficient')
parser.add_argument('--V_theta', type=float, default=5.0,
                    help=' Dynamic threshold weighting coefficient')
parser.add_argument('--alpha_U', type=float, default=0.023,
                    help=' Internal activity decay coefficient')
parser.add_argument('--V_U', type=int, default=1,
                    help=' Internal activity weighting coefficient')
parser.add_argument('--t', type=int, default=60,
                    help=' Number of iterations for RCNN ignition')
parser.add_argument('--sigma_kernel', type=int, default=4,
                    help=' Variance of 2-D Gaussian distribution for Gaussian kernel matrix')
parser.add_argument('--sigma_random_closure', type=int, default=5,
                    help=' Variance of 2-D Gaussian distribution for random closure probability matrix')
parser.add_argument('--size', type=int, default=9,
                    help=' Gaussian kernel size (size by size)')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')

# reinitialize the model
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=150,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-04
                    ,help='learning rate')
parser.add_argument('--decay', type=str, default='100',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*SmoothL1Loss',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')

args = parser.parse_args()


args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
