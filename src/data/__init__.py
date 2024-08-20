from importlib import import_module
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset

class MyConcatDataset(ConcatDataset):

    # Merge multiple dataset objects into one dataset object.
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

#  The MyConcatDataset class is used to merge multiple dataset objects into a single dataset object. It also provides a method called set_scale() to set the scale for the datasets within the merged dataset
class Data:
    '''
    This class is used for Initializing loader
    '''
    def __init__(self, args):
        self.loader_train = None

        if not args.test_only:
            datasets = []

            # If the RCNN channel is enabled, the generated dataset folder will be named according to specific rules
            if args.RCNN_channel == 'on':
                for d in args.data_train:
                    module_name = d + '_RCNN' if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                    m = import_module('data.' + module_name.lower())
                    datasets.append(getattr(m, module_name)(args, name=d))
            else:
                for d in args.data_train:
                    module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                    m = import_module('data.' + module_name.lower())
                    datasets.append(getattr(m, module_name)(args, name=d))

            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)

            #     getattr(x, 'y') is equivalent to x.y
            else:
                if args.RCNN_channel == 'on':
                    module_name = d + '_RCNN' if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                else:
                    module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)
                print(testset)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )