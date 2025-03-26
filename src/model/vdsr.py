from model import common

import torch.nn as nn
import torch.nn.init as init
def make_model(args, parent=False):
    return VDSR(args)
class VDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(VDSR, self).__init__()

        
        self.input_channels = args.n_colors      # 假设为 1

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 

        # 定义残差块
        def basic_block(in_channels, out_channels, act):
            return common.BasicBlock(
                conv, in_channels, out_channels, kernel_size,
                bias=True, bn=False, act=act
            )

        # body 部分保持输入输出通道一致
        m_body = []
        m_body.append(basic_block(self.input_channels, n_feats, nn.ReLU(True)))
        for _ in range(n_resblocks - 2):
            m_body.append(basic_block(n_feats, n_feats, nn.ReLU(True)))
        m_body.append(basic_block(n_feats, self.input_channels, None))  # 输出通道与输入一致

        self.body = nn.Sequential(*m_body)



    def forward(self, x):
        # x: (1, 2, 196, 196)
        res = self.body(x)      # res: (1, 2, 196, 196)
        res += x                # 残差连接，形状匹配   
        return res

