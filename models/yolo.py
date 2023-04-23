# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    #
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes                       输入的种类
        self.no = nc + 5  # number of outputs per anchor        种类+xywh置信度
        self.nl = len(anchors)  # number of detection layers    anchors数3 anchors=[[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.na = len(anchors[0]) // 2  # number of anchors      na = 6/2 = 3
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid [tensor([]), tensor([]), tensor([])]
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid [tensor([]), tensor([]), tensor([])]
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        #ModuleList(
        #    (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
        #    (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
        #    (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
        #)
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl): #self.nl=3 anchors=[[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
            x[i] = self.m[i](x[i])  # conv (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))传入x[i]也就是最后输入的3个数据矩阵
            # ModuleList(
            #    (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
            #    (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
            #    (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
            # )
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            #重新整形.view(8, 2) transpose为转换变量，例如 正常为permute（0，1，2，3，4）总共0-4列转为permute(0, 1, 3, 4, 2) .contiguous()为copy
            #【1，3，85，32，32】self.na 一个cell的anchor数 self.no种类+xywh置信度
            #before：torch.Size([1, 255, 32, 32])经过1*1的卷积，得到255通道数 也就是说 255 别下边的分割成了3个anchor数与85个特征数量
            #torch.Size([1, 3, 32, 32, 85]) 1是对这3个特征矩阵进行编号 3是每个cell的3个anchor数 32*32是总共那么多的cell box nx是xyzh 置信度 + 种类
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    #nx 32 ny 32 I
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    #归一化然后 输入的种类+1（1是conf）0 ((2, 2, 81), 4)
                    # .split（2,4）4是切割维度，2是将这个维度每2个进行划分  .split（(2, 2, 81),4）是将第4维度的数据进行2分 2分 81分进行切割
                    # tensor([[[[[0.50794, 0.47921],
                    #            [0.50794, 0.47921],
                    #            [0.50794, 0.47921],
                    #            ...,
                    #            [0.50794, 0.47921],
                    #            [0.50794, 0.47921],
                    #            [0.50794, 0.47921]],
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    #中心点坐标和宽高信息从特征图上映射到原图上self.stride存储了由正向传播中计算出的缩放倍数，然后再乘以缩放倍数，就回到原图上了
                    #实际上的公式是y =（2.0 * x +（绝对坐标- 0.5）） * 缩放倍数 self.grid[i] = （绝对坐标- 0.5） self.stride[i] = 缩放倍数
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))
        #如果训练则直接返回x 如果不是训练那么就是(torch.cat(z, 1), x)
        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        #【1，3，32，32，2】
        shape = 1, self.na, ny, nx, 2  # grid shape
        #tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        #yv tensor([[0., 0., 0., ..., 0., 0., 0.],
        #         [1., 1., 1., ..., 1., 1., 1.],
        #         [2., 2., 2., ..., 2., 2., 2.],
        #         ...,
        #         [29., 29., 29., ..., 29., 29., 29.],
        #         [30., 30., 30., ..., 30., 30., 30.],
        #         [31., 31., 31., ..., 31., 31., 31.]])
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        # tensor([[[[[0., 0.],
        #            [1., 0.],
        #            [2., 0.],
        #            ...,
        #            [29., 0.],
        #            [30., 0.],
        #            [31., 0.]],
        #           [[0., 1.],
        #            [1., 1.],
        #            [2., 1.],
        #  torch.stack((xv, yv), 2).expand(shape) 为每个x y坐标计算出他们的绝对位置，由相对变为绝对坐标 比如第一个是（1，0）的坐标代表第一行的第2个cell 加上相对的坐标就是绝对的位置
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    # YOLOv5 base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    # YOLOv5 detection model 模型的加载yaml
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        #cfg为配置文件 cfg是不是dict类型
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict 如果是，直接加载
        else:  # is *.yaml 如果是文件
            import yaml  # for torch hub 导入yaml
            self.yaml_file = Path(cfg).name #路径
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict 加载模型

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels 读取Ch通道，yaml中并没有这个值，所以默认为3 并添一个ch
        if nc and nc != self.yaml['nc']: #如果传入的nc与yaml文件的nc并不相同，则将yaml文件中的nc覆盖掉
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:#如果传入anchors与yaml文件的anchors不相同，则将yaml文件的anchors覆盖掉
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        #加载所有模型的框架 model ， save参数【3，32，6，2，2】
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        #创建一个每一类别的长度的列表
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        #inplace
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors 用于构造最后一层的stride
        m = self.model[-1]  # Detect() 取出最后一层
        if isinstance(m, (Detect, Segment)):#判断最后一层是否是Detect层
            s = 256  # 2x min stride
            m.inplace = self.inplace
            #进行一次前馈传播 使用m是Segment是的话则 de forward（x）：self.forward(x)[0] 否则返回 de forward（x）：self.forward(x)
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            #创建了一张空白图片 list(Tensor(1,3,32,32,85),Tensor(1,3,16,16,85)),Tensor(1,3,8,8,85))
            forward_ = forward(torch.zeros(1, ch, s, s))
            #
            m.stride = torch.tensor([s / x.shape[-2] for x in forward_])  # forward [8 16 32] 根据输入一张图片根据图片输出的尺寸，算出Detect的步长
            #
            check_anchor_order(m)
            #
            m.anchors /= m.stride.view(-1, 1, 1)
            #步长
            self.stride = m.stride
            #
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None


def parse_model(d, ch):  # model_dict, input_channels(3) d = .yaml ch=[3] 处理模型
    # Parse a YOLOv5 model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    #从yaml文件中取出anchors nc depth_multiple width_multiple activation
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    #anchors [10,13, 16,30, 33,23] 6个值 //2就是3个anchors
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    #nc是类别 ：d['nc'] na是classes 5就是x y w h no就是输出的矩阵维度
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5) 255
    #layers创建的每一层 save统计那些层特征图是需要保存的 c2 每一层的维度输出
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    #d['backbone']骨干网络   d['head']头部网络i, (f, n, m, args) i是序号 (f, n, m, args)是[from, number, module, args]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        #eval将变量的名称(String)转为该变量(object)算出写入表达式的结果 isinstance(m, str)如果是字符串则转为类
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain gd=depth_multiple round(number * depth_multiple)
        #为1这 直接就是1 conv (c1 输入维度, c2输出维度, k=1卷积核, s=1步长, p=None 边缘padding, g=1, d=1, act=True):
        if m in {
                Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
            #c1表示输入维度 c2标识输出维度 [from, number, module, args]
            c1, c2 = ch[f], args[0]# c1：3 c2：64
            #如果c2输出维度并不是no
            if c2 != no:  # if not output [from, number, module, args] args[0] 是不是最终输出的通道倍数 这有bug
                c2 = make_divisible(c2 * gw, 8) #c2 * gw = 64*0.5 math.ceil(32 / 8) * 8
            #[输入维度，输出维度, args[1:]] 【3，32，6，2，2】
            args = [c1, c2, *args[1:]]
            #如果在这几个，输入变量需要增加一个n n为这层卷积需要重复几次
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                #c1, c2, n = 1需要插入n, shortcut = True, g = 1
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            #对于f中的每个x，ch[x]都会被加入到列表中
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        # [输入维度，输出维度, args[1:]] 【3，32，6，2，2】
        #m(*args) m( 【3，32，6，2，2】)
        #如果 n 大于 1，那么就使用 m(*args) 重复 n 次，然后将这些重复的结果作为参数传递给 nn.Sequential。如果 n 不大于 1，那么就直接使用 m(*args) 作为参数传递给 nn.Sequential。12
        #一般处理c3模块，处理多个c3模块
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        #  replace函数作用是字符串"__main__"替换为''，在当前项目没有用到这个替换
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        #参数总量 = 卷积核长 * 卷积核宽 * 输入维度 * 输出维度（卷积核数量 = 输入维度 * 输出维度）
        #参数总量 = 卷积核长 * 卷积核宽 * 卷积核数量
        #x.numel() for x in m_.parameters() 比那里所有层的参数
        np = sum(x.numel() for x in m_.parameters())  # number params
        #(conv): Conv2d(3, 32, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2), bias=False)
        #    (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (act): SiLU()
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        #那些层是需要保存的 判断form
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)  #save是那些要保存的层数[6 4 14 10]然后排序


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
