import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

# If the parameters in args have been adjusted, running this file will start model training,
# but it is still recommended to run demo.sh for parameter adjustment in subsequent experiments

def main():
    global model

    # Determine the type of processing task(video or image)
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()

    else:
        if checkpoint.ok:

            # Initialize loader, model and loss
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None

            # Encapsulate the above initialization components into the training process
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():

                # Call two functions in the trainer separately to initiate the training and testing processes
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':

    main()
