import os
import sys
import math
from decimal import Decimal
from ptflops import get_model_complexity_info
import utility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.option import args
import torch
import torch.nn.utils as utils
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet18
from thop import profile
from thop.vision.basic_hooks import zero_ops
from fvcore.nn import FlopCountAnalysis, parameter_count


def reduce_to_14bit(data):
    """
    将 16bit 数据降低为 14bit，同时保持数据类型为 float32。
    """
    # 将数据缩放到 14bit 范围
    data = data * (2**14 - 1) / (2**16 - 1)
    return data

def restore_to_16bit(data):
    """
    将 14bit 数据恢复为 16bit，同时保持数据类型为 float32。
    """
    # 将数据恢复到 16bit 范围
    data = data * (2**16 - 1) / (2**14 - 1)
    return data

class Trainer():
    """
    This class is used to encapsulate the training and testing process of the model
    """
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_last_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            
            

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)


            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch() + 1
        torch.save(self.model.state_dict(), '/data/project/3DEDSR/MODEL/model_{}.pt'.format(epoch))
        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )

        self.model.eval()

        timer_test = utility.timer()

        self.args.save_results = True
        if self.args.save_results: self.ckp.begin_background()

        # Start background processes to handle the task of saving results
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)

                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    # 将 lr 从 16bit 变更为 14bit
                    # lr = reduce_to_14bit(lr)
                    sr = self.model(lr, idx_scale)
                    # 将 sr 从 14bit 恢复为 16bit
                    # sr = restore_to_16bit(sr)
                    sr = utility.quantize(sr, args.rgb_range)

                    save_list = [sr]
                    now_psnr = utility.calc_psnr(
                        sr, hr, scale, rgb_range=args.rgb_range, dataset=d
                    )
                    self.ckp.log[-1, idx_data, idx_scale] += now_psnr

                    # # 保存 sr 和 hr 图像
                    # save_filename = os.path.splitext(filename[0])[0]  # 去掉文件扩展名
                    # sr_save_path = os.path.join(save_dir, f"{save_filename}_sr.tif")
                    # hr_save_path = os.path.join(save_dir, f"{save_filename}_hr.tif")
                    
                    # # 将 sr 和 hr 转换为 NumPy 数组并调整数据范围
                    # sr_np = sr.squeeze().cpu().numpy() # 转换为 [0, 65535]
                    # hr_np = hr.squeeze().cpu().numpy() # 转换为 [0, 65535]
                    
                    # # 转换为 16-bit 数据类型
                    # sr_np = sr_np.astype(np.uint16)
                    # hr_np = hr_np.astype(np.uint16)
            
                    # # 保存为 16-bit TIFF 格式
                    # Image.fromarray(sr_np).save(sr_save_path)
                    # Image.fromarray(hr_np).save(hr_save_path)
                    
                    if self.args.save_gt:
                        save_list.extend([lr, hr])
                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)

                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        if self.args.cpu:
            device = torch.device('cpu')
        else:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device(self.args.cuda)  # 确保 self.args.cuda 是一个字符串
            else:
                device = torch.device('cpu')

        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]
    
    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

