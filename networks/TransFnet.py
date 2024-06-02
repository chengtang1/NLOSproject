import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from utils import *
# transformlayer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from einops.layers.torch import Rearrange

class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, embed_dim=96,
                 depths=[2, 2, 6], num_heads=[3, 6, 12],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, **kwargs):

        super().__init__()

        patches_resolution = [img_size // patch_size, img_size // patch_size]
        num_patches = patches_resolution[0] * patches_resolution[1]

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=None)
            self.layers.append(layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
# fuse the cnn and transformer feature in the Fourier domain(due to the consideration of frequency)

# class FFM(nn.Module):
#     def __init__(self, F_trans=None, F_cnn=None):
#         super().__init__()
#
#     def forward(self,F_trans, F_cnn):
#         F_trans_fft = torch.fft.fftn(F_trans,dim=1,norm="backward")
#         F_cnn_fft = torch.fft.ifftn(F_cnn,dim=1,norm="backward")
#         F_fusion = F_trans_fft+F_cnn_fft
#         F_fusion = torch.fft.ifftn(F_fusion)
#         F_fusion = F_fusion.real+F_fusion.imag
#         # 乘以共轭
#         # F_fusion = F_fusion*F_fusion.conjucate()
#         return F_fusion
class FF(nn.Module):
    def __init__(self,f1=None,f2=None,f3=None,f4=None,channel=None):
        super().__init__()
        c1 = f1.shape[1]
        c2 = f2.shape[1]
        c = channel
        self.conv_1 = nn.Conv2d(c1, c, kernel_size=1)
        self.conv_2 = nn.Conv2d(c2, c, kernel_size=1,padding=0,stride=2,dilation=1)
        self.conv_3 = nn.Conv2d(c, c, kernel_size=1,padding=0,stride=2,dilation=1)
    def forward(self, f1,f2,f3,f4):
        f1,f2,f3,f4 = torch.fft.fft2(f1),torch.fft.fft2(f2),torch.fft2(f3),torch.fft.fft2(f4)
        F_cat = torch.cat((torch.cat((f1,f2),dim=2),torch.cat((f3,f4),dim=2)),dim=3)
        #  iuput both into the convBlock
        F_cat_1 = self.conv_1(F_cat)
        # im conv operation
        F_cat_2 = self.conv_2(F_cat_1)
        F_cat_2 = torch.fft.ifft(F_cat_2)
        F_cat_2 = F_cat_2*torch.conj(F_cat_2)  #1x768x7x7
        return F_cat_2


class transfnet(nn.Module):

    def __init__(self, num_classes, in_channels=1, initial_filter_size=96, kernel_size=3, do_instancenorm=True):
        super().__init__()
        # initial the transformer and FFM  通过 checkpoint 加载imagnet 预训练的参数

        model_path = r'./weights/swin_tiny_patch4_window7_224.pth'   # 加载路径参数
        self.swin_transformer = SwinTransformer(224, in_chans=3)
        checkpoint = torch.load(model_path, map_location=torch.device(device))['model']
        unexpected = ["patch_embed.proj.weight", "patch_embed.proj.bias", "patch_embed.norm.weight",
                      "patch_embed.norm.bias",
                      "head.weight", "head.bias", "layers.0.downsample.norm.weight", "layers.0.downsample.norm.bias",
                      "layers.0.downsample.reduction.weight", "layers.1.downsample.norm.weight",
                      "layers.1.downsample.norm.bias",
                      "layers.1.downsample.reduction.weight", "layers.2.downsample.norm.weight",
                      "layers.2.downsample.norm.bias",
                      "layers.2.downsample.reduction.weight", "norm.weight", "norm.bias"]

        for key in list(checkpoint.keys()):
            if key in unexpected or 'layers.3' in key:
                del checkpoint[key]
        self.swin_transformer.load_state_dict(checkpoint)

        self.FFM = FF()
        #调整的卷积层
        # self.reshape_conv_1 = nn.Conv2d(448,96,kernel_size=1)
        # self.reshape_conv_1_inverse = nn.Conv2d(96,448,kernel_size=1)
        # initial the layers
        self.contr_1_1 = self.contract(in_channels, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.contr_1_2 = self.contract(initial_filter_size, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.contr_2_1 = self.contract(initial_filter_size, initial_filter_size*2, kernel_size, instancenorm=do_instancenorm)
        self.contr_2_2 = self.contract(initial_filter_size*2, initial_filter_size*2, kernel_size, instancenorm=do_instancenorm)
        # self.pool2 = nn.MaxPool2d(2, stride=2)

        self.contr_3_1 = self.contract(initial_filter_size*2, initial_filter_size*2**2, kernel_size, instancenorm=do_instancenorm)
        self.contr_3_2 = self.contract(initial_filter_size*2**2, initial_filter_size*2**2, kernel_size, instancenorm=do_instancenorm)
        # self.pool3 = nn.MaxPool2d(2, stride=2)

        self.contr_4_1 = self.contract(initial_filter_size*2**2, initial_filter_size*2**3, kernel_size, instancenorm=do_instancenorm)
        self.contr_4_2 = self.contract(initial_filter_size*2**3, initial_filter_size*2**3, kernel_size, instancenorm=do_instancenorm)
        # self.pool4 = nn.MaxPool2d(2, stride=2)

        self.center = nn.Sequential(
            nn.Conv2d(initial_filter_size*2**3, initial_filter_size*2**4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filter_size*2**4, initial_filter_size*2**4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(initial_filter_size*2**4, initial_filter_size*2**3, 2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.expand_4_1 = self.expand(initial_filter_size*2**4, initial_filter_size*2**3)
        self.expand_4_2 = self.expand(initial_filter_size*2**3, initial_filter_size*2**3)
        self.upscale4 = nn.ConvTranspose2d(initial_filter_size*2**3, initial_filter_size*2**2, kernel_size=2, stride=2)

        self.expand_3_1 = self.expand(initial_filter_size*2**3, initial_filter_size*2**2)
        self.expand_3_2 = self.expand(initial_filter_size*2**2, initial_filter_size*2**2)
        self.upscale3 = nn.ConvTranspose2d(initial_filter_size*2**2, initial_filter_size*2, 2, stride=2)

        self.expand_2_1 = self.expand(initial_filter_size*2**2, initial_filter_size*2)
        self.expand_2_2 = self.expand(initial_filter_size*2, initial_filter_size*2)
        self.upscale2 = nn.ConvTranspose2d(initial_filter_size*2, initial_filter_size, 2, stride=2)

        self.expand_1_1 = self.expand(initial_filter_size*2, initial_filter_size)
        self.expand_1_2 = self.expand(initial_filter_size, initial_filter_size)
        # Output layer for segmentation
        self.final = nn.Conv2d(initial_filter_size, num_classes, kernel_size=1)  # kernel size for final layer = 1, see paper

        self.softmax = torch.nn.Softmax2d()

        # Output layer for "autoencoder-mode"
        self.output_reconstruction_map = nn.Conv2d(initial_filter_size, out_channels=1, kernel_size=1)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, instancenorm=True):
        if instancenorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.LeakyReLU(inplace=True),
            )
        return layer

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def forward(self, x, enable_concat=True, print_layer_shapes=False):
       # 设置concat的权重比例
        concat_weight = 1
        if not enable_concat:
            concat_weight = 0
        b,h,w = x.shape[0],x.shape[1],x.shape[2]
        x = torch.reshape(x,(b,1,h,w))
        contr_1 = self.contr_1_2(self.contr_1_1(x))
        pool = self.pool(contr_1)   # bx96x56x56

        fm1_reshaped = Rearrange('b c h w -> b (h w) c')(pool)    # input bx96x56x56 input the transformer layer
        pool_embed = self.swin_transformer.layers[0](fm1_reshaped)
        pool = Rearrange('b (h w) c -> b c h w',h=h//2, w=w//2)(pool_embed)

        contr_2 = self.contr_2_2(self.contr_2_1(pool))
        pool = self.pool(contr_2)   # bx192x28x28

        fm1_reshaped = Rearrange('b c h w -> b (h w) c')(pool)  # input bx192x28x28 input the transformer layer
        pool_embed = self.swin_transformer.layers[1](fm1_reshaped)
        pool = Rearrange('b (h w) c -> b c h w', h=h // 4, w=w // 4)(pool_embed)



        contr_3 = self.contr_3_2(self.contr_3_1(pool))
        pool = self.pool(contr_3)  # bx384x14x14

        fm1_reshaped = Rearrange('b c h w -> b (h w) c')(pool)  # input bx192x28x28 input the transformer layer
        pool_embed = self.swin_transformer.layers[2](fm1_reshaped)
        pool_trans = Rearrange('b (h w) c -> b c h w', h=h // 8, w=w // 8)(pool_embed)

        pool_fusion = self.FFM(pool, pool_trans)

        contr_4 = self.contr_4_2(self.contr_4_1(pool_fusion))
        pool = self.pool(contr_4) # 64x1x8x8


        center = self.center(pool)

        crop = self.center_crop(contr_4, center.size()[2], center.size()[3])
        concat = torch.cat([center, crop*concat_weight], 1)

        expand = self.expand_4_2(self.expand_4_1(concat))
        upscale = self.upscale4(expand)

        crop = self.center_crop(contr_3, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop*concat_weight], 1)

        expand = self.expand_3_2(self.expand_3_1(concat))
        upscale = self.upscale3(expand)

        crop = self.center_crop(contr_2, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop*concat_weight], 1)

        expand = self.expand_2_2(self.expand_2_1(concat))
        upscale = self.upscale2(expand)

        crop = self.center_crop(contr_1, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop*concat_weight], 1)

        expand = self.expand_1_2(self.expand_1_1(concat))

        if enable_concat:
            output = self.final(expand)
        if not enable_concat:
            output = self.output_reconstruction_map(expand)
        return output
