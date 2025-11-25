from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging

from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from lib.models.DCNv2.dcn_v2 import DCN
from lib.utils.decode import _nms, _topk_N
import matplotlib.pyplot as plt
import numpy as np

import cv2

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Decoder3D(nn.Module):
    def __init__(self, in_channels=16, out_channels=16):
        """
        扩展版本：增加额外卷积层，进一步提升特征融合能力
        """
        super(Decoder3D, self).__init__()
        self.deconv1 = nn.ConvTranspose3d(in_channels, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose3d(32, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.relu2 = nn.ReLU(inplace=True)

        self.deconv3 = nn.ConvTranspose3d(32, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        输入: (B, 16, T, H, W)
        输出: (B, 16, T, H, W)
        """
        x = self.relu1(self.bn1(self.deconv1(x)))
        x = self.relu2(self.bn2(self.deconv2(x)))
        x = self.relu3(self.bn3(self.deconv3(x)))
        H, W = x.size(3), x.size(4)
        x = F.adaptive_max_pool3d(x, output_size=(1, H, W))
        x = x.permute(0, 2, 1, 3, 4).squeeze(1)
        return x


class BasicConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)  # verify bias false
        self.bn = nn.BatchNorm3d(out_channels,
                                 eps=0.001,  # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TemporalFeatureExtractor3D(nn.Module):
    def __init__(self, in_channels):
        """
        Args:
            in_channels (int): 输入tensor的通道数.
        输出:
            (B, 16, T, H, W) - 保持T, H, W不变，通道数变为16.
        """
        super(TemporalFeatureExtractor3D, self).__init__()

        # 分支1：侧重时序信息，采用3个沿时间轴的卷积层
        self.branch1 = nn.Sequential(
            BasicConv3d(in_channels=in_channels, out_channels=16,
                        kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            BasicConv3d(in_channels=16, out_channels=16,
                        kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            BasicConv3d(in_channels=16, out_channels=16,
                        kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        )

        # 分支2：侧重空间信息，采用3个在H、W方向上的卷积层
        self.branch2 = nn.Sequential(
            BasicConv3d(in_channels=in_channels, out_channels=16,
                        kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            BasicConv3d(in_channels=16, out_channels=16,
                        kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            BasicConv3d(in_channels=16, out_channels=16,
                        kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        )

        # 分支3：捕捉联合时空信息，采用3个3D卷积层
        self.branch3 = nn.Sequential(
            BasicConv3d(in_channels=in_channels, out_channels=16,
                        kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            BasicConv3d(in_channels=16, out_channels=16,
                        kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            BasicConv3d(in_channels=16, out_channels=16,
                        kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        )

        # 融合层：将三个分支输出拼接（总通道数 16*3=48）后，通过1×1×1卷积压缩到16通道
        self.fuse = BasicConv3d(in_channels=48, out_channels=16,
                                kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

    def forward(self, x):
        """
        Args:
            x: 输入tensor，尺寸为 (B, C, T, H, W)
        Returns:
            输出tensor，尺寸为 (B, 16, T, H, W)
        """
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        # 按通道维度拼接三个分支的输出
        out = torch.cat([out1, out2, out3], dim=1)
        # 融合后调整通道数至16
        out = self.fuse(out)
        return out



class MaskGuidedFusionModel(nn.Module):
    def __init__(self, threshold=0.3, channel_sizes=[16, 32, 64, 128, 256]):
        super(MaskGuidedFusionModel, self).__init__()
        self.threshold = threshold

        # 注意力权重生成模块
        self.att_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * ch, ch, 1),
                nn.Sigmoid()
            ) for ch in channel_sizes
        ])

        # 特征融合模块
        self.fuse_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            ) for ch in channel_sizes
        ])

        # 归一化层
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(ch) for ch in channel_sizes
        ])

    def forward(self, displacement_map, pre_features, features):
        fused_features = []

        # 首帧处理：使用零初始化作为伪前一帧特征
        if pre_features is None:
            pre_features = [torch.zeros_like(f) for f in features]

        # 验证特征维度匹配
        assert len(pre_features) == len(features), \
            f"特征尺度数量不匹配：pre_features({len(pre_features)}) vs features({len(features)})"

        for idx, (pre_feat, curr_feat) in enumerate(zip(pre_features, features)):
            warped_feat = warp_feature_with_flow(pre_feat, displacement_map)

            concat_feat = torch.cat([warped_feat, curr_feat], dim=1)


            att = self.att_conv[idx](concat_feat)  # 注意力权重
            fused = self.fuse_conv[idx](concat_feat)  # 融合特征

            output = curr_feat + att * fused
            output = self.bn_layers[idx](output)

            fused_features.append(output)

        return fused_features

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)


    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x, temporal_feat):
        y = []
        x = self.base_layer(x)
        x = x + temporal_feat
        for i in range(5):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights, strict=False)


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')

    return model


def warp_feature_with_flow(feature, flow):
    B, C, h, w = feature.shape
    B, _, H, W = flow.shape  # 原始 flow 的尺寸

    # 1. 将 flow 下采样到当前特征尺寸 (h, w)
    flow_resized = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=True)

    # 2. 调整 flow 数值：根据尺寸缩放因子，使得位移数值适配低分辨率
    flow_resized[:, 0, :, :] *= (w / W)  # x 方向
    flow_resized[:, 1, :, :] *= (h / H)  # y 方向

    # 3. 构造基础采样网格
    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    base_grid = torch.stack((grid_x, grid_y), dim=-1).float().to(feature.device)
    base_grid = base_grid.unsqueeze(0).expand(B, h, w, 2)

    sampling_grid = base_grid - flow_resized.permute(0, 2, 3, 1)

    sampling_grid_norm = sampling_grid.clone()
    sampling_grid_norm[..., 0] = 2.0 * sampling_grid[..., 0] / (w - 1) - 1.0
    sampling_grid_norm[..., 1] = 2.0 * sampling_grid[..., 1] / (h - 1) - 1.0

    warped_feature = F.grid_sample(feature, sampling_grid_norm, mode='bilinear', align_corners=True)
    return warped_feature



def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        # Adjusting the output channels to be 2 * in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, padding=1)  # Output 2 * out_channels
        self.bn2 = nn.BatchNorm2d(out_channels * 2)

        # A skip connection for residual block, output channels = 2 * out_channels
        self.skip = nn.Conv2d(in_channels, out_channels * 2, kernel_size=1)  # Match output channels

    def forward(self, x):
        residual = self.skip(x)  # Skip connection with matching channels
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Adding residual connection
        return out


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x



class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)

            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            # layers[i] = project(layers[i])
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            # print(i, channels[j], in_channels[j:])
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]
        for i in range(len(layers) - self.startp - 1):  # 0, 1, 2, 3
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])  # 4, 3, 2, 1
        return out


class DLASeg(nn.Module):
    def __init__(self, heads, radius_mapping, final_kernel,
                 head_conv, out_channel=0):
        super(DLASeg, self).__init__()
        self.prev_state = False
        self.first_level = 0  # int(np.log2(down_ratio))
        self.last_level = 3  # last_level
        self.backbone = dla34(pretrained=True)

        self.mask_guided_fusion = MaskGuidedFusionModel()
        self.conv_3d = TemporalFeatureExtractor3D(3)
        self.deconv_3d = Decoder3D(16, 16)

        self.mask_layer = nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=True)
        self.mask_layer.bias.data.fill_(-4.6)
        self.bn = nn.BatchNorm2d(16)
        self.radius_mapping = radius_mapping

        channels = [16, 32, 64, 128, 256]  # self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]  # 1, 2, 4, 8, 16
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)
        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(channels[self.first_level], head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=final_kernel, stride=1,
                              padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-4.6)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def decode_mask_to_objects(self, mask, K=450, score_threshold=0.08):
        """
        将mask解码为目标对象信息，返回不带梯度的张量

        Args:
            mask: 掩码特征图，形状为(B, C, H, W)
            K: 提取的目标数量，默认为450
            score_threshold: 目标置信度阈值，默认为0.08

        Returns:
            obj_info: 目标信息张量，形状为(B, K, 5)，每行包含[score, x, y, ind, cls]
                      对于置信度低于阈值的目标，所有值均为-1
        """
        with torch.no_grad():
            # 应用sigmoid
            sigmoid_mask = torch.sigmoid(mask)

            # 应用非最大抑制
            nms_mask = _nms(sigmoid_mask)

            # 解码得到目标信息
            topk_scores, topk_inds, topk_clses, topk_ys, topk_xs, cls_inds_masks = _topk_N(nms_mask, K=K, num_classes=1)

            # 整理成张量格式 (B, K, 5)
            obj_info = torch.zeros((topk_scores.size(0), K, 5), device=topk_scores.device)

            # 填充obj_info张量
            obj_info[:, :, 0] = topk_scores  # score
            obj_info[:, :, 1] = topk_xs  # x
            obj_info[:, :, 2] = topk_ys  # y
            obj_info[:, :, 3] = topk_inds  # ind
            obj_info[:, :, 4] = topk_clses  # cls

            # 对于置信度不大于阈值的目标，将所有信息都设为-1
            invalid_mask = topk_scores <= score_threshold
            obj_info[invalid_mask, :] = -1

        return obj_info  # 返回时已经不含梯度

    def train_feat(self, x, c0, i, displacement_map=None, pre_frame_features=None):
        B, C, H, W = x.shape

        if displacement_map == None:
            displacement_map = torch.zeros((B, 2, H, W), dtype=torch.float32, device=x.device)
        current_frame_features = self.backbone(x, c0[:, i])
        mask_fused_feats = self.mask_guided_fusion(displacement_map, pre_frame_features, current_frame_features)
        p0, p1, p2, p3, _ = self.dla_up(mask_fused_feats)
        y = [p0, p1, p2]
        self.ida_up(y, 0, len(y))
        p0 = y[-1]

        return current_frame_features, p0, p1, p2, p3

    def track_feat(self, x, c0, i, N, displacement_map=None, pre_frame_features=None):
        B, _, H, W = x.shape

        if self.prev_state != False and i < N - 1:
            return self.prev_feat[i + 1][0], self.prev_feat[i + 1][1], self.prev_feat[i + 1][2], self.prev_feat[i + 1][
                3], self.prev_feat[i + 1][4]
        else:
            if displacement_map == None:
                displacement_map = torch.zeros((B, 2, H, W), dtype=torch.float32, device=x.device)
            current_frame_features = self.backbone(x, c0[:, i])
            mask_fused_feats = self.mask_guided_fusion(displacement_map, pre_frame_features, current_frame_features)
            p0, p1, p2, p3, _ = self.dla_up(mask_fused_feats)
            y = [p0, p1, p2]
            self.ida_up(y, 0, len(y))
            p0 = y[-1]

            return current_frame_features, p0, p1, p2, p3

    def flow_driven_motion_estimator(self, curr_frame, next_frame, obj_info, r):

        B, C, H, W = curr_frame.shape
        device = curr_frame.device

        # 反归一化图像
        mean = torch.tensor([0.49965, 0.49965, 0.49965], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.08255, 0.08255, 0.08255], device=device).view(1, 3, 1, 1)

        curr_frame_orig = curr_frame * std + mean
        next_frame_orig = next_frame * std + mean


        curr_gray_cpu = (curr_frame_orig.cpu().numpy() * 255).astype(np.uint8)
        next_gray_cpu = (next_frame_orig.cpu().numpy() * 255).astype(np.uint8)

        displacement_map = torch.zeros((B, 2, H, W), dtype=obj_info.dtype, device=device)

        curr_points_map = torch.zeros((B, 2, H, W), dtype=obj_info.dtype, device=device)

        next_point_map = torch.zeros((B, 3, H, W), dtype=torch.float32, device=device)
        next_point_map[:, 0, :, :].fill_(-1)

        for b in range(B):
            # 获取当前批次的图像，转置为 (H, W, 3)
            curr_img = curr_gray_cpu[b].transpose(1, 2, 0)
            next_img = next_gray_cpu[b].transpose(1, 2, 0)

            curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(next_img, cv2.COLOR_RGB2GRAY)

            # 获取当前批次有效目标的坐标
            valid_mask = obj_info[b, :, 0] > 0  # 置信度大于0的目标
            valid_points = obj_info[b, valid_mask, 1:3]  # 取x,y坐标
            valid_points_np = valid_points.cpu().numpy()

            if valid_points_np.shape[0] > 0:
                # 格式化为OpenCV的输入格式，注意转换为 (N, 1, 2) 的形状
                prev_pts = valid_points_np.astype(np.float32).reshape(-1, 1, 2)

                next_pts, status, err = cv2.calcOpticalFlowPyrLK(curr_gray, next_gray, prev_pts, None)
                next_pts = next_pts.squeeze(1)
                status = status.squeeze(1)
                next_pts_tensor = torch.from_numpy(next_pts).to(device)  # (num_valid, 2)
                status_tensor = torch.from_numpy(status).to(device)  # (num_valid,)

                # 遍历每个有效目标点，使用光流匹配出的坐标作为保存位移的索引
                for i, (orig_x, orig_y) in enumerate(valid_points_np):
                    # 计算光流匹配得到的点位置，并取整作为索引
                    tracked_x = int(round(next_pts_tensor[i, 0].item()))
                    tracked_y = int(round(next_pts_tensor[i, 1].item()))
                    if 0 <= tracked_x < W and 0 <= tracked_y < H:
                        status_val = status_tensor.item() if status_tensor.dim() == 0 else status_tensor[i].item()
                        # 在索引位置（由next_pts确定）保存匹配状态和跟踪坐标
                        next_point_map[b, 0, tracked_y, tracked_x] = status_val
                        next_point_map[b, 1, tracked_y, tracked_x] = next_pts_tensor[i, 0]
                        next_point_map[b, 2, tracked_y, tracked_x] = next_pts_tensor[i, 1]
                        # 在同一位置保存原始目标点坐标（方便后续计算位移差）
                        curr_points_map[b, 0, tracked_y, tracked_x] = float(orig_x)
                        curr_points_map[b, 1, tracked_y, tracked_x] = float(orig_y)

        # 构造一个跟踪成功的掩码（状态为1表示跟踪成功）
        success_mask = (next_point_map[:, 0, :, :] == 1)
        # 计算 x 和 y 方向的位移差
        disp_x = next_point_map[:, 1, :, :] - curr_points_map[:, 0, :, :]
        disp_y = next_point_map[:, 2, :, :] - curr_points_map[:, 1, :, :]

        # 对于跟踪成功的位置，赋值位移；否则保持为0
        displacement_map[:, 0, :, :] = disp_x * success_mask.float()
        displacement_map[:, 1, :, :] = disp_y * success_mask.float()

        # 对每个跟踪成功的点，将其位移赋值给以该点为中心的5x5邻域
        # 这里我们遍历每个batch
        for b in range(B):
            # 找到成功跟踪的点的索引
            success_idx = torch.nonzero(success_mask[b], as_tuple=False)
            for idx in success_idx:
                y, x = idx.tolist()  # y, x位置
                # 计算邻域边界（确保不越界）
                y0 = max(y - r, 0)
                y1 = min(y + r, H)
                x0 = max(x - r, 0)
                x1 = min(x + r, W)
                # 将该位置的位移扩展到邻域内
                displacement_map[b, 0, y0:y1, x0:x1] = disp_x[b, y, x]
                displacement_map[b, 1, y0:y1, x0:x1] = disp_y[b, y, x]

        return displacement_map

    def forward(self, img_input, training=True, vid=None):
        # x : B, N, C, H, W
        # ..., -3, -2, -1, 0
        _, N, _, _, _ = img_input.shape

        ret_temp = {}

        temp_feat = []
        final_map = None
        pre_frame_features = None
        temporal_input = img_input.permute(0, 2, 1, 3, 4)
        c0 = self.conv_3d(temporal_input)
        c0_input = c0.permute(0, 2, 1, 3, 4)

        for i in range(N):
            x = img_input[:, i, :]
            # Extract current frame features using backbone
            if training:
                pre_features, p0, p1, p2, p3 = self.train_feat(x, c0_input, i, final_map, pre_frame_features)
                temp_feat.append([pre_features, p0, p1, p2, p3])
            else:
                pre_features, p0, p1, p2, p3 = self.track_feat(x, c0_input, i, N, final_map, pre_frame_features)
                temp_feat.append([pre_features, p0, p1, p2, p3])

            pre_frame_features = pre_features

            # mask prop
            if i == 0:
                mask_out = self.mask_layer(p0).unsqueeze(1)  # B, N, 1, 512, 512
            else:
                mask_out = torch.cat((mask_out, self.mask_layer(p0).unsqueeze(1)), dim=1)

            # 使用封装的函数解码mask
            current_mask = mask_out[:, i]  # B, 1, H, W
            obj_info = self.decode_mask_to_objects(current_mask, K=700, score_threshold=0.1)

            # 如果不是最后一帧，执行光流跟踪（无论是训练还是推理模式）
            if i < N - 1:
                # 处理当前帧和下一帧
                final_map = self.flow_driven_motion_estimator(img_input[:, i], img_input[:, i + 1], obj_info, self.radius_mapping)
            else:
                final_map = None
            # construct sequence
            if i == 0:
                p0_seq = p0.unsqueeze(1)
            else:
                p0_seq = torch.cat((p0_seq, p0.unsqueeze(1)), dim=1)  # B, N, 16, 512, 512

        ret_temp['hm_seq'] = mask_out
        p0_seq = p0_seq.permute(0, 2, 1, 3, 4)
        final_hm = self.deconv_3d(p0_seq)

        if not training:
            self.prev_feat = temp_feat
            self.prev_state = True

        ret = {}
        for head in self.heads:
            ret_temp[head] = getattr(self, head)(final_hm)  ##其他的头reg，wh，hm，是利用reconstructed_current作为输入
        ret[1] = ret_temp
        return [temp_feat, ret]


def dla_dcn_net(heads, radius_mapping, head_conv=128):
    model = DLASeg(heads, radius_mapping, final_kernel=1,
                   head_conv=head_conv)
    return model