import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# only implements ResNext bottleneck c


# """This strategy exposes a new dimension, which we call “cardinality”
# (the size of the set of transformations), as an essential factor
# in addition to the dimensions of depth and width."""
CARDINALITY = 32
DEPTH = 4
BASEWIDTH = 64


# """The grouped convolutional layer in Fig. 3(c) performs 32 groups
# of convolutions whose input and output channels are 4-dimensional.
# The grouped convolutional layer concatenates them as the outputs
# of the layer."""

class ResNextBottleNeckC(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        
        C = CARDINALITY  # How many groups a feature map was splitted into
        
        # """We note that the input/output width of the template is fixed as
        # 256-d (Fig. 3), We note that the input/output width of the template
        # is fixed as 256-d (Fig. 3), and all widths are dou- bled each time
        # when the feature map is subsampled (see Table 1)."""
        D = int(DEPTH * out_channels / BASEWIDTH)  # number of channels per group
        self.split_transforms = nn.Sequential(
            nn.Conv2d(in_channels, C * D, kernel_size=1, groups=C, bias=False),
            nn.BatchNorm2d(C * D),
            nn.ELU(inplace=True),
            nn.Conv2d(C * D, C * D, kernel_size=3, stride=stride, groups=C, padding=1, bias=False),
            nn.BatchNorm2d(C * D),
            nn.ELU(inplace=True),
            nn.Conv2d(C * D, out_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
        )
        
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )
    
    def forward(self, x):
        return F.elu(self.split_transforms(x) + self.shortcut(x))


class ResNext(nn.Module):
    
    def __init__(self, block, num_blocks, class_names=10):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True)
        )
        
        self.conv2 = self._make_layer(block, num_blocks[0], 64, 1)
        self.conv3 = self._make_layer(block, num_blocks[1], 128, 2)
        self.conv4 = self._make_layer(block, num_blocks[2], 256, 2)
        self.conv5 = self._make_layer(block, num_blocks[3], 512, 2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, class_names)
        # Top layer
        self.toplayer = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.drop=nn.Dropout(p=0.5)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 128, kernel_size=1, stride=1, padding=0)

        self.trans_conv=nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 2, stride=1, padding=1, output_padding=0, groups=1, bias=True, dilation=1),
            torch.nn.ConvTranspose2d(64, 16, 2, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1),
            torch.nn.ConvTranspose2d(16, 3, 1, stride=1, padding=0, output_padding=0, groups=1, bias=True,
                                     dilation=1)
        )

        self.reduce_tanh = nn.Sequential(
            nn.Tanh()
        )

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):

        x1f = self.conv1(x)
        f = torch.einsum('nihw,njhw->nij', x1f, x1f)
        f = torch.einsum('nij,nihw->njhw', f, x1f) / (x1f.size(2) * x1f.size(3)) + x1f

        c2f = self.conv2(f)
        f = torch.einsum('nihw,njhw->nij', c2f, c2f)
        f = torch.einsum('nij,nihw->njhw', f, c2f) / (c2f.size(2) * c2f.size(3)) + c2f

        c3f = self.conv3(f)

        c4f = self.conv4(c3f)

        c5f = self.conv5(c4f)

        p5 = self.toplayer(c5f)
        p4 = self._upsample_add(p5, self.latlayer1(c4f))
        p3 = self._upsample_add(p4, self.latlayer2(c3f))
        p2 = self._upsample_add(p3, self.latlayer3(c2f))
        # Smooth

        p2 = self.smooth3(p2)

        p2 = self.trans_conv(p2)

        f = torch.einsum('nihw,njhw->nij', x, x)
        f = torch.einsum('nij,nihw->njhw', f, x) / (p2.size(2) * p2.size(3)) + p2

        x1 = self.conv1(f)
        f = torch.einsum('nihw,njhw->nij', x1, x1)
        f = torch.einsum('nij,nihw->njhw', f, x1) / (x1.size(2) * x1.size(3)) + x1

        c2 = self.conv2(f)
        f = torch.einsum('nihw,njhw->nij', c2, c2)
        f = torch.einsum('nij,nihw->njhw', f, c2) / (c2.size(2) * c2.size(3)) + c2

        c3 = self.conv3(f)

        c4 = self.conv4(c3)

        c5 = self.conv5(c4)

        x2 = self.avg(c5)
        x2 = x2.view(x2.size(0), -1)
        class_pre = self.fc(self.drop(x2))
        class_pre = self.reduce_tanh(class_pre)

        return p2, class_pre
    def _make_layer(self, block, num_block, out_channels, stride):
        """Building resnext block
        Args:
            block: block type(default resnext bottleneck c)
            num_block: number of blocks per layer
            out_channels: output channels per block
            stride: block stride

        Returns:
            a resnext layer
        """
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * 4
        
        return nn.Sequential(*layers)


def resnext50():
    """ return a resnext50(c32x4d) network
    """
    return ResNext(ResNextBottleNeckC, [3, 4, 6, 3])


def resnext101():
    """ return a resnext101(c32x4d) network
    """
    return ResNext(ResNextBottleNeckC, [3, 4, 23, 3])


def resnext152():
    """ return a resnext101(c32x4d) network
    """
    return ResNext(ResNextBottleNeckC, [3, 4, 36, 3])


def test():
    n=resnext50()
    fms = n(torch.autograd.Variable(torch.randn(1,3,64,64)))
    for fm in fms:
        print(fm.size())

