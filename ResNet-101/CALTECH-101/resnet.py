import torch.nn as nn
import math
from collections import OrderedDict
import torch.nn.functional as F
import torch


def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = conv3x3(inplanes, planes, stride)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['elu'] = nn.ELU(inplace=True)
        m['conv2'] = conv3x3(planes, planes)
        m['bn2'] = nn.BatchNorm2d(planes)
        self.group1 = nn.Sequential(m)

        self.elu= nn.Sequential(nn.ELU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual

        out = self.elu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        m  = OrderedDict()
        m['conv1'] = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['elu1'] = nn.ELU(inplace=True)
        m['conv2'] = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        m['bn2'] = nn.BatchNorm2d(planes)
        m['elu2'] = nn.ELU(inplace=True)
        m['conv3'] = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        m['bn3'] = nn.BatchNorm2d(planes * 4)
        self.group1 = nn.Sequential(m)

        self.elu= nn.Sequential(nn.ELU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual
        out = self.elu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=102):
        self.inplanes = 64
        super(ResNet, self).__init__()

        m = OrderedDict()
        m['conv1'] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m['bn1'] = nn.BatchNorm2d(64)
        m['elu1'] = nn.ELU(inplace=True)
        m['maxpool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group1= nn.Sequential(m)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.Sequential(nn.AvgPool2d(7))
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.drop=nn.Dropout(p=0.5)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        self.trans_conv=nn.Sequential(
            torch.nn.ConvTranspose2d(256, 64, 2, stride=2, padding=0, output_padding=0, groups=1, bias=True, dilation=1),
            torch.nn.ConvTranspose2d(64, 16, 2, stride=2, padding=0, output_padding=0, groups=1, bias=True, dilation=1),
            torch.nn.ConvTranspose2d(16, 3, 1, stride=1, padding=0, output_padding=0, groups=1, bias=True,
                                     dilation=1)
        )


        self.group2 = nn.Sequential(
            OrderedDict([
                ('fc', nn.Linear(512 * block.expansion, num_classes))
            ])
        )

        self.reduce_tanh=nn.Sequential(
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


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
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        x1f = self.group1(x)
        f= torch.einsum('nihw,njhw->nij',x1f,x1f)
        f= torch.einsum('nij,nihw->njhw',f, x1f)/(x1f.size(2)*x1f.size(3))+x1f

        c2f = self.layer1(f)
        f= torch.einsum('nihw,njhw->nij',c2f,c2f)
        f= torch.einsum('nij,nihw->njhw',f, c2f)/(c2f.size(2)*c2f.size(3))+c2f

        c3f = self.layer2(f)


        c4f = self.layer3(c3f)


        c5f = self.layer4(c4f)

        p5 = self.toplayer(c5f)
        p4 = self._upsample_add(p5, self.latlayer1(c4f))
        p3 = self._upsample_add(p4, self.latlayer2(c3f))
        p2 = self._upsample_add(p3, self.latlayer3(c2f))
        # Smooth

        p2 = self.smooth3(p2)
    
        p2=self.trans_conv(p2)

        f= torch.einsum('nihw,njhw->nij',x,x)
        f= torch.einsum('nij,nihw->njhw',f, x)/(p2.size(2)*p2.size(3))+p2

        x1 = self.group1(f)
        f= torch.einsum('nihw,njhw->nij',x1,x1)
        f= torch.einsum('nij,nihw->njhw',f, x1)/(x1.size(2)*x1.size(3))+x1

        c2 = self.layer1(f)
        f= torch.einsum('nihw,njhw->nij',c2,c2)
        f= torch.einsum('nij,nihw->njhw',f, c2)/(c2.size(2)*c2.size(3))+c2
        
        c3 = self.layer2(f)
 
        c4 = self.layer3(c3)
   
        c5 = self.layer4(c4)

        x2 = self.avgpool(c5)
        x2 = x2.view(x2.size(0), -1)
        class_pre = self.group2(self.drop(x2))
        class_pre =self.reduce_tanh(class_pre)

        return p2,class_pre


def resnet18(pretrained=False, model_root=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    return model


def resnet34(pretrained=False, model_root=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    return model


def resnet50(pretrained=False, model_root=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def resnet101(pretrained=False, model_root=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def resnet152(pretrained=False, model_root=None, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    return model



def test():
    n=resnet101()
    fms = n(torch.autograd.Variable(torch.randn(1,3,224,224)))
    for fm in fms:
        print(fm.size())

