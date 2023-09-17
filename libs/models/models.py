import math
from typing import Any, Callable
import timm

import torch
import torch.nn as nn

from .cam import GradCAM
from .fix_weight_dict import fix_model_state_dict

import torch.nn.functional as F


def weights_init(init_type="gaussian") -> Callable:
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find("Conv") == 0 or classname.find("Linear") == 0) and hasattr(
            m, "weight"
        ):
            if init_type == "gaussian":
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == "default":
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class BENet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3) -> None:
        super(BENet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(128, out_channels), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_maxpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 相当于下采样
class Cvi(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        before: str = None,
        after: str = None,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super(Cvi, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.after: Any[Callable]
        self.before: Any[Callable]
        self.conv.apply(weights_init("gaussian"))
        if after == "BN":
            self.after = nn.BatchNorm2d(out_channels)
        elif after == "Tanh":
            self.after = torch.tanh
        elif after == "sigmoid":
            self.after = torch.sigmoid

        if before == "ReLU":
            self.before = nn.ReLU(inplace=True)
        elif before == "LReLU":
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if hasattr(self, "before"):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, "after"):
            x = self.after(x)

        return x

# 上采样
class CvTi(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        before: str = None,
        after: str = None,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super(CvTi, self).__init__()
        self.after: Any[Callable]
        self.before: Any[Callable]
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, bias
        )
        self.conv.apply(weights_init("gaussian"))
        if after == "BN":
            self.after = nn.BatchNorm2d(out_channels)
        elif after == "Tanh":
            self.after = torch.tanh
        elif after == "sigmoid":
            self.after = torch.sigmoid

        if before == "ReLU":
            self.before = nn.ReLU(inplace=True)
        elif before == "LReLU":
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if hasattr(self, "before"):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, "after"):
            x = self.after(x)

        return x

#####################################new##################

class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, norm=None, act=None) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.norm = nn.Identity() if norm is None else norm(out_channels)
        self.act = nn.Identity() if act is None else act()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x) 
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm=None, act=None):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, norm=norm, act=act)
        self.conv2 = ConvNormAct(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, norm=norm, act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, norm=None, act=None):
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv = BasicBlock(in_channels=mid_channels, out_channels=out_channels, norm=norm, act=act)

    def forward(self, x, cat):
        x = self.up(x)
        x = torch.cat([x, cat], dim=1)
        x = self.conv(x)
        return x
    


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, channels=[64, 128, 256, 512, 1024], norm=nn.InstanceNorm2d, act=nn.ReLU) -> None:
        super().__init__()
        self.enc1 = BasicBlock(in_channels, channels[0], norm=norm, act=act)
        self.enc2 = BasicBlock(channels[0], channels[1], norm=norm, act=act)
        self.enc3 = BasicBlock(channels[1], channels[2], norm=norm, act=act)
        self.enc4 = BasicBlock(channels[2], channels[3], norm=norm, act=act)
        self.enc5 = BasicBlock(channels[3], channels[4], norm=norm, act=act)
        self.down = nn.MaxPool2d(2)
        self.dec4 = UpBlock(channels[4], channels[3], norm=norm, act=act)
        self.dec3 = UpBlock(channels[3], channels[2], norm=norm, act=act)
        self.dec2 = UpBlock(channels[2], channels[1], norm=norm, act=act)
        self.dec1 = UpBlock(channels[1], channels[0], norm=norm, act=act)
        self.out_conv = nn.Conv2d(channels[0], num_classes, 1, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x = self.down(x1)
        x2 = self.enc2(x)
        x = self.down(x2)
        x3 = self.enc3(x)
        x = self.down(x3)
        x4 = self.enc4(x)
        x = self.down(x4)
        x = self.enc5(x)
        x = self.dec4(x, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)
        x = self.out_conv(x)
        return x
    

class Generator_best(nn.Module):
    def __init__(self, in_channels=7, num_classes=3, norm=nn.InstanceNorm2d, act=nn.ReLU) -> None:
        super().__init__()

        backbone = timm.create_model('seresnet50', pretrained=False)
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            backbone.bn1,
            backbone.act1,
            
        )
        self.down = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.dec4 = UpBlock(2048, 1024, norm=norm, act=act)
        self.dec3 = UpBlock(1024, 512, norm=norm, act=act)
        self.dec2 = UpBlock(512, 256, norm=norm, act=act)
        self.dec1 = UpBlock(256, 64, 128, norm=norm, act=act)
        self.dec0 = UpBlock(64, 32, 32+in_channels)
        self.out_conv = nn.Conv2d(32, num_classes, 1, 1)
    

    def forward(self, inputs):
        x1 = self.layer0(inputs)
        x = self.down(x1)
        x2= self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x = self.layer4(x4)
        x = self.dec4(x, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)
        x = self.dec0(x, inputs)
        x = self.out_conv(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation, downsample, norm=None, act=None, BOTTLENECK_EXPANSION=4):
        super(Bottleneck, self).__init__()
        mid_channels = out_channels // BOTTLENECK_EXPANSION
        self.reduce = ConvNormAct(in_channels, mid_channels, 1, stride, 0, 1, norm=norm, act=act)
        self.conv3x3 = ConvNormAct(mid_channels, mid_channels, 3, 1, dilation, dilation, norm=norm, act=act)
        self.increase = ConvNormAct(mid_channels, out_channels, 1, 1, 0, 1, norm=norm, act=act)

        self.shortcut = (
            ConvNormAct(in_channels, out_channels, 1, stride, 0, 1, norm=norm, act=nn.Identity)
            if downsample
            else nn.Identity()
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)

class ResBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, out_channels, stride, dilation, multi_grids=None):
        super(ResBlock, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(num_layers)]
        else:
            assert num_layers == len(multi_grids)

        for i in range(num_layers):
            self.add_module(
                "block{}".format(i + 1),
                Bottleneck(
                    in_channels=(in_channels if i == 0 else out_channels),
                    out_channels=out_channels,
                    stride=(stride if i == 0 else 1),
                    dilation=dilation * multi_grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )


class Pool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = ConvNormAct(in_channels, out_channels, 1, 1, 0, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", ConvNormAct(in_channels, out_channels, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                ConvNormAct(in_channels, out_channels, 3, 1, padding=rate, dilation=rate),
            )
        self.stages.add_module("imagepool", Pool(in_channels, out_channels))

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)


class Generator1(nn.Sequential):
    def __init__(self, in_channels=7, num_classes=3, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18], multi_grids=[1, 2, 4], output_stride=8, input_size=(256, 256)):
        super(Generator, self).__init__()

        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        ch = [64 * 2 ** p for p in range(6)]
        self.add_module("layer1", nn.Sequential(
            ConvNormAct(in_channels, ch[0], 7, 2, 3, 1),
            nn.MaxPool2d(3, 2, 1, ceil_mode=True)))

        self.add_module("layer2", ResBlock(n_blocks[0], ch[0], ch[2], s[0], d[0]))
        self.add_module("layer3", ResBlock(n_blocks[1], ch[2], ch[3], s[1], d[1]))
        self.add_module("layer4", ResBlock(n_blocks[2], ch[3], ch[4], s[2], d[2]))
        self.add_module(
            "layer5", ResBlock(n_blocks[3], ch[4], ch[5], s[3], d[3], multi_grids)
        )
        self.add_module("aspp", ASPP(ch[5], 256, atrous_rates))
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.add_module("fc1", ConvNormAct(concat_ch, 256, 1, 1, 0, 1))
        self.add_module("fc2", nn.Conv2d(256, num_classes, kernel_size=1))
        self.add_module("upsample", nn.UpsamplingBilinear2d(size=input_size))

#####################################end##################


###############################小波####################
def dwt_init(x):
    x01 = x[:, :, 0::2, :] /2
    x02 = x[:, :, 1::2, :] /2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

###############################end####################



#######################new2#############################
from pdb import set_trace as stx
import numbers
import timm

from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)



        attn1 = (v @ k.transpose(-2, -1)) * self.temperature
        attn1 = attn1.softmax(dim=-1)

        out1 = (attn1 @ q)

        out = (attn @ v)
        out = out1 + out

        

        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
class Generator(nn.Module):
    def __init__(self, 
        inp_channels=3,
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Generator, self).__init__()

        # self.Generator = Generator(inp_channels, num_classes=3, norm=nn.InstanceNorm2d, act=nn.ReLU)
        # self.conv = nn.Conv2d(inp_channels,3,kernel_size=1)
        self.dwt=DWT()
        self.iwt=IWT()


        self.patch_embed = OverlapPatchEmbed(inp_channels*4, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), inp_channels*4, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        # inp_img =self.Generator(inp_img)
        # inp_img = self.conv(inp_img)
        inp_img = self.dwt(inp_img)

        inp_enc_level1 = self.patch_embed(inp_img)
        
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return self.iwt(out_dec_level1)









#######################end#############################







####################################### old ##############
# start
# class Generator(nn.Module):  # in_channels 7  out_channels 3
#     def __init__(self, in_channels: int = 7, out_channels: int = 3) -> None:
#         super(Generator, self).__init__()

#         self.Cv0 = Cvi(in_channels, 64)

#         self.Cv1 = Cvi(64, 128, before="LReLU", after="BN")

#         self.Cv2 = Cvi(128, 256, before="LReLU", after="BN")

#         self.Cv3 = Cvi(256, 512, before="LReLU", after="BN")

#         self.Cv4 = Cvi(512, 512, before="LReLU", after="BN")

#         self.Cv5 = Cvi(512, 512, before="LReLU")

#         self.CvT6 = CvTi(512, 512, before="ReLU", after="BN")

#         self.CvT7 = CvTi(1024, 512, before="ReLU", after="BN")

#         self.CvT8 = CvTi(1024, 256, before="ReLU", after="BN")

#         self.CvT9 = CvTi(512, 128, before="ReLU", after="BN")

#         self.CvT10 = CvTi(256, 64, before="ReLU", after="BN")

#         self.CvT11 = CvTi(128, out_channels, before="ReLU", after="Tanh")

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         # encoder
#         x0 = self.Cv0(input)
#         x1 = self.Cv1(x0)
#         x2 = self.Cv2(x1)
#         x3 = self.Cv3(x2)
#         x4_1 = self.Cv4(x3)
#         x4_2 = self.Cv4(x4_1)
#         x4_3 = self.Cv4(x4_2)
#         x5 = self.Cv5(x4_3)

#         # decoder
#         x6 = self.CvT6(x5)

#         cat1_1 = torch.cat([x6, x4_3], dim=1)
#         x7_1 = self.CvT7(cat1_1)
#         cat1_2 = torch.cat([x7_1, x4_2], dim=1)
#         x7_2 = self.CvT7(cat1_2)
#         cat1_3 = torch.cat([x7_2, x4_1], dim=1)
#         x7_3 = self.CvT7(cat1_3)

#         cat2 = torch.cat([x7_3, x3], dim=1)
#         x8 = self.CvT8(cat2)

#         cat3 = torch.cat([x8, x2], dim=1)
#         x9 = self.CvT9(cat3)

#         cat4 = torch.cat([x9, x1], dim=1)
#         x10 = self.CvT10(cat4)

#         cat5 = torch.cat([x10, x0], dim=1)
#         out = self.CvT11(cat5)

#         return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=6) -> None:
        super(Discriminator, self).__init__()

        self.Cv0 = Cvi(in_channels, 64)

        self.Cv1 = Cvi(64, 128, before="LReLU", after="BN")

        self.Cv2 = Cvi(128, 256, before="LReLU", after="BN")

        self.Cv3 = Cvi(256, 512, before="LReLU", after="BN")

        self.Cv4 = Cvi(512, 1, before="LReLU", after="sigmoid")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x0 = self.Cv0(input)
        x1 = self.Cv1(x0)
        x2 = self.Cv2(x1)
        x3 = self.Cv3(x2)
        out = self.Cv4(x3)

        return out

# end


def benet(pretrained: bool = False, **kwargs: Any) -> BENet:
    model = BENet(**kwargs)  # {'in_channels': 3}
    if pretrained:  # 默认开启了预训练
        state_dict = torch.load("./pretrained/pretrained_benet.prm")  # map_location
        model.load_state_dict(fix_model_state_dict(state_dict))
    return model


def cam_benet(pretrained: bool = False, **kwargs: Any) -> GradCAM:
    model = BENet(**kwargs)
    if pretrained:
        state_dict = torch.load("./pretrained/pretrained_benet.prm")  # map_location
        model.load_state_dict(fix_model_state_dict(state_dict))
    model.eval()  # BENet用的就是预训练模型，不参与更新了
    target_layer = model.features[3]  # shape: 
    wrapped_model = GradCAM(model, target_layer)  # 拿一层去可视化
    return wrapped_model


def generator(pretrained: bool = False, **kwargs: Any) -> Generator:
    model = Generator(**kwargs)
    if pretrained:
        state_dict = torch.load("./pretrained/pretrained_g_srnet.prm")
        model.load_state_dict(fix_model_state_dict(state_dict))
    return model


def discriminator(pretrained: bool = False, **kwargs: Any) -> Discriminator:
    model = Discriminator(**kwargs)
    if pretrained:
        state_dict = torch.load("./pretrained/pretrained_d_srnet.prm")
        model.load_state_dict(fix_model_state_dict(state_dict))
    return model


if __name__ == '__main__':
    import torchsummary
    model = Generator().cuda()
    
    torchsummary.summary(model, (7, 512, 512), 2, 'cuda')