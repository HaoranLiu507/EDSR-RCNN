from model import common
import torch
import torch.nn as nn

# Pre-trained models with different parameters can be downloaded from the following URL,
# where r represents the number of residual blocks, f represents the number of feature maps,
# and the suffix number represents the reconstruction magnification



def make_model(args, parent=False):
    return EDSR_RCNN(args)

class AdaptiveFusionHead(nn.Module):
    def __init__(self, img_channels, n_feats, kernel_size):
        super().__init__()
        self.img_channels = img_channels
        self.img_conv = common.default_conv(img_channels, n_feats, kernel_size)
        self.rcnn_conv = common.default_conv(1, n_feats, kernel_size)
        self.fusion = nn.Sequential(
            common.default_conv(2*n_feats, n_feats, 1),
            nn.ReLU(inplace=True)
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            common.default_conv(2*n_feats, 2*n_feats//16, 1),
            nn.ReLU(inplace=True),
            common.default_conv(2*n_feats//16, 2*n_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        img = x[:, :self.img_channels, :, :]
        rcnn = x[:, self.img_channels:, :, :]
        
        img_feat = self.img_conv(img)
        rcnn_feat = self.rcnn_conv(rcnn)
        
        # Channel attention for adaptive fusion
        combined = torch.cat([img_feat, rcnn_feat], dim=1)
        weights = self.channel_attention(combined)
        weighted_feat = combined * weights
        
        fused = self.fusion(weighted_feat)
        return fused


class AdaptiveSKFusion(nn.Module):
    """
    Adaptive SKFusion module to fuse the regular image channels with an additional RCNN channel.
    It processes the image part and the RCNN extra channel separately using convolution,
    then fuses the features using an adaptive, selective kernel mechanism.
    """
    def __init__(self, n_colors, out_channels, kernel_size=3, reduction=16, conv=common.default_conv):
        super(AdaptiveSKFusion, self).__init__()
        # Process the normal image channels (n_colors)
        self.conv_img = conv(n_colors, out_channels, kernel_size)
        # Process the extra RCNN channel (assumed to be 1 channel)
        self.conv_rcnn = conv(1, out_channels, kernel_size)
        # Global pooling and FC layers to compute selective attention weights
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            conv(out_channels, out_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            conv(out_channels // reduction, 2 * out_channels, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)
        self.out_channels = out_channels
        self.n_colors = n_colors

    def forward(self, x):
        # x shape: (batch, n_colors + 1, H, W)
        # Split the input into the regular image and RCNN extra channel
        img = x[:, :self.n_colors, :, :]
        rcnn = x[:, self.n_colors:, :, :]
        # Get features for each branch
        feat_img = self.conv_img(img)
        feat_rcnn = self.conv_rcnn(rcnn)
        # Compute a combined feature map for attention estimation
        feat_sum = feat_img + feat_rcnn
        attn = self.global_pool(feat_sum)
        attn = self.fc(attn)
        # Reshape to have two attention maps (one per branch)
        attn = attn.view(-1, 2, self.out_channels, 1, 1)
        attn = self.softmax(attn)
        # Fuse features using the adaptive weights
        out = feat_img * attn[:, 0] + feat_rcnn * attn[:, 1]
        return out


class EDSR_RCNN(nn.Module):
    """
    This class is used to initialize the EDSR network structure proposed in the article
    """
    def __init__(self, args, conv=common.default_conv):
        super(EDSR_RCNN, self).__init__()

        n_resblocks = args.n_resblocks # Number of residual blocks
        n_feats = args.n_feats # Number of feature maps
        kernel_size = 3 # Convolutional layer kernel size
        scale = args.scale[0]
        act = nn.ReLU(True)

        SUPPORTED_HEADS = {
        "adaptive": AdaptiveFusionHead,
        "SKFusion": AdaptiveSKFusion,
        "Convolution": conv
        }

        # If the RCNN channel is enabled, the input convolutional layer will expand by one dimension
        if args.RCNN_channel == "on":
            # The input tensor will have shape (batch, n_colors+1, H, W).
            # We use the adaptive SKFusion module to fuse the extra channel with the image channels.
            m_head_model = SUPPORTED_HEADS[args.model_head]
            m_head = [m_head_model(args.n_colors, n_feats, kernel_size)]
        else:
            m_head = [conv(args.n_colors, n_feats, kernel_size)]

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
