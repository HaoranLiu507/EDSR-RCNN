import os
import optuna
import torch
from model import common
import utility
import data
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from src.option import args
import time
import torch.nn.utils as utils
import numpy as np

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = utility.checkpoint(args)


def make_optimizer(args, target, lr=1e-5):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_last_lr(self):
            return self.scheduler.get_last_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer


# Optimize Model with Optuna
class EDSR(nn.Module):
    def __init__(self, op_resblocks, op_feats, op_res_scale, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = op_resblocks
        n_feats = op_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        if args.RCNN_channel == "on":
            channels = args.n_colors + 1
        else:
            channels = args.n_colors
        # define head module
        m_head = [conv(channels, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=op_res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)


        return x

# This function is used to optimize parameters such as learning rate to achieve better results for the model
def objective(trial):
    epoch_op = args.epochs
    op_re = trial.suggest_int("op_resblocks", 30, 40, step=5)
    op_fe = trial.suggest_int("op_feats", 250, 400, step=10)
    op_res = trial.suggest_float("op_rescale", 0.1, 0.3, step=0.1)
    model_optuna = EDSR(op_re, op_fe, op_res).to(device)
    lr_star = trial.suggest_float("lr_star", 1e-6, 1e-4, log=True)
    optimizer = make_optimizer(args, model_optuna, lr=lr_star)
    gclip_flag = trial.suggest_categorical("gclip_flag", [True, False])
    gclip = trial.suggest_float("gclip", 0.5, 1, step=0.1)
    opyunaloss = nn.SmoothL1Loss()

    loader = data.Data(args)
    scale_op = args.scale
    dir_op = os.path.join('..', 'experiment_log_jilu', 'config256.txt')
    time_start = time.time()
    for epoch in range(epoch_op + 1):
        torch.set_grad_enabled(True)
        model_optuna.train()
        lr_last = optimizer.get_last_lr()
        print("Lr in epoch{}:{}".format(epoch, lr_last))
        for batch, (lr, hr, _,) in enumerate(loader.loader_train):
            lr = lr.to(device)
            hr = hr.to(device)
            optimizer.zero_grad()
            sr = model_optuna(lr)
            loss = opyunaloss(sr, hr)
            loss.backward()
            if gclip_flag is False:
                pass
            else:
                utils.clip_grad_value_(
                    model_optuna.parameters(),
                    gclip
                )
            optimizer.step()
        optimizer.schedule()

        torch.set_grad_enabled(False)
        psnr_store = []

        model_optuna.eval()
        with torch.no_grad():
            for idx_data, d in enumerate(loader.loader_test):
                for idx_scale, scale in enumerate(scale_op):
                    d.dataset.set_scale(idx_scale)
                    for lr, hr, filename in d:
                        lr = lr.to(device)
                        hr = hr.to(device)
                        sr = model_optuna(lr)
                        sr = utility.quantize(sr, rgb_range=args.rgb_range)
                        print("test sr image:{}".format(sr))
                        psnr_optuna = 0
                        psnr_optuna += utility.calc_psnr(
                            sr, hr, scale, rgb_range=args.rgb_range, dataset=d
                        )
                        print("PSNR in epoch{}:{}".format(epoch, psnr_optuna))
                        psnr_store.append(psnr_optuna)
        if epoch == epoch_op:
            torch.save(model_optuna.state_dict(), 'model_r{}_trial{}.pt'.format(args.rgb_range, trial.number))
        psnr_mean = sum(psnr_store) / len(psnr_store)
        time_end = time.time()
        time_code = time_end - time_start
        print("PSNR_mean in epoch{}:{}".format(epoch, psnr_mean))
        with open(dir_op, 'a') as f:
            f.write("code running time:{} s".format(time_code))
            f.write('\n')
            f.write("trial_number:{}".format(trial.number))
            f.write("PSNR in epoch{}:{}".format(epoch, psnr_mean))
            f.write('\n')
            f.write("LOSS in epoch{}:{}".format(epoch, loss))
            f.write('\n')
        trial.report(psnr_mean, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return psnr_mean


if __name__ == "__main__":
    storage_name = "sqlite:///rcnn_edsr_r256.db"
    if os.path.exists("rcnn_edsr_r256.db"):
        print("study resumed")
        study_load = optuna.load_study(storage=storage_name, study_name="rcnn_edsr_r256.db")
        study_load.optimize(objective, n_trials=10, timeout=None)
    else:
        print("study created")
        study = optuna.create_study(direction="maximize",
                                    pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
                                    study_name="rcnn_edsr_r256.db",
                                    storage=storage_name,
                                    load_if_exists=True,
                                    sampler=optuna.samplers.RandomSampler()
                                    )
        study.optimize(objective, n_trials=10, timeout=None)
