from model import common

import torch.nn as nn

# Pre-trained models with different parameters can be downloaded from the following URL,
# where r represents the number of residual blocks, f represents the number of feature maps,
# and the suffix number represents the reconstruction magnification
url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}


def make_model(args, parent=False):
    return EDSR(args)


class EDSR(nn.Module):
    """
    This class is used to initialize the EDSR network structure proposed in the article
    """
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks # Number of residual blocks
        n_feats = args.n_feats # Number of feature maps
        kernel_size = 3 # Convolutional layer kernel size
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None

        # If the RCNN channel is enabled, the input convolutional layer will expand by one dimension
        if args.RCNN_channel == "on":
            channels = args.n_colors + 1
        else:
            channels = args.n_colors

        # define head module(input convolutional layer)
        m_head = [conv(channels, n_feats, kernel_size)]

        # define body module(Residual block structure)
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module(Upsampling module and output convolutional layer)
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats,args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    # Encapsulate the three parts mentioned above into a residual structure
    def forward(self, x):

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
