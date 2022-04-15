import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from secml_malware.models.basee2e import End2EndModel


class MalConv(nn.Module):
    "keep Embed layer for traning"
    def __init__(self, max_input_size=102400, window_size=500,vocab_size=257):
        super(MalConv, self).__init__()

        self.embed = nn.Embedding(vocab_size, 8, padding_idx=0)    # [1,256], 0,in total 257

        self.conv_1 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)

        self.pooling = nn.MaxPool1d(int(max_input_size / window_size))


        self.fc_1 = nn.Linear(128,128)
        self.fc_2 = nn.Linear(128,2)

        self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax()


    def forward(self,x):
        x = self.embed(x.long())
        # Channel first
        x = torch.transpose(x,-1,-2)

        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))

        x = cnn_value * gating_weight
        x = self.pooling(x)

        x = x.view(-1,128)
        x = self.fc_1(x)
        x = self.fc_2(x)
        # x = self.sigmoid(x)

        return x


class MalConv_freezeEmbed(nn.Module):
    "produce adve_x: freeze the embed layer, to produce aadversarial example"

    def __init__(self, max_input_size=102400, window_size=500,vocab_size=257):
        super(MalConv_freezeEmbed, self).__init__()

        self.embed = nn.Embedding(vocab_size, 8, padding_idx=0)  # [1,256], 0,in total 257

        self.conv_1 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)

        self.pooling = nn.MaxPool1d(int(max_input_size / window_size))

        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128, 2)

        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()

    def forward(self, x):
        # x = self.embed(x.long())
        # Channel first
        x = torch.transpose(x, -1, -2)  # shape [1,102400,8] --> [1,8,102400]

        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))

        x = cnn_value * gating_weight
        x = self.pooling(x)

        x = x.view(-1, 128)
        x = self.fc_1(x)
        x = self.fc_2(x)
        # x = self.sigmoid(x)

        return x



class FireEye(nn.Module):
    """
    DNN model from "Activation Analysis of a Byte-Based Deep Neural Network for Malware Classification" FireEye Inc.
    keep embed layer for training
    """
    def __init__(self, input_length=102400, window_size=512,vocab_size=257):
        super(FireEye, self).__init__()

        self.embed = nn.Embedding(vocab_size, 10, padding_idx=0)

        self.conv_1 = nn.Conv1d(10, 16, 8, stride=4, bias=True)
        self.conv_2 = nn.Conv1d(16, 32, 16, stride=4, bias=True)
        self.conv_3 = nn.Conv1d(32, 64, 4, stride=2, bias=True)
        self.conv_4 = nn.Conv1d(64, 128, 4, stride=2, bias=True)
        self.conv_5 = nn.Conv1d(128, 128, 4, stride=2, bias=True)


        self.pooling = nn.MaxPool1d(int(input_length / window_size))
        # self.pooling = nn.AdaptiveMaxPool1d(1)

        # self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128*3, 2)

        self.sigmoid = nn.Sigmoid() # binary classification
        # self.softmax = nn.Softmax()   # multi-classfication

    def forward(self, x):
        x = self.embed(x.long())
        # Channel first
        x = torch.transpose(x, -1, -2)

        x = self.conv_1(x)
        # x = self.pooling(x)
        x = self.conv_2(x)
        # x = self.pooling(x)
        x = self.conv_3(x)
        # x = self.pooling(x)
        x = self.conv_4(x)
        # x = self.pooling(x)
        x = self.conv_5(x)
        x = self.pooling(x)

        x = x.view(x.size(0), -1)
        x = self.fc_2(x)

        return x



class FireEye_freezeEmbed(nn.Module):
    """
    freeze embed layer for adversarial example generation

    remove embed layer in forward process, calculated embedded results first and then input it to the model
    """

    def __init__(self, input_length=102400, window_size=512,vocab_size=257):
        super(FireEye_freezeEmbed, self).__init__()

        self.embed = nn.Embedding(vocab_size, 10, padding_idx=0)

        self.conv_1 = nn.Conv1d(10, 16, 8, stride=4, bias=True)
        self.conv_2 = nn.Conv1d(16, 32, 16, stride=4, bias=True)
        self.conv_3 = nn.Conv1d(32, 64, 4, stride=2, bias=True)
        self.conv_4 = nn.Conv1d(64, 128, 4, stride=2, bias=True)
        self.conv_5 = nn.Conv1d(128, 128, 4, stride=2, bias=True)


        self.pooling = nn.MaxPool1d(int(input_length / window_size))
        # self.pooling = nn.AdaptiveMaxPool1d(1)

        # self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128 * 3, 2)
        self.sigmoid = nn.Sigmoid() # binary classification
        # self.softmax = nn.Softmax()   # multi-classfication

    def forward(self, x):
        # x = self.embed(x.long())
        # Channel first
        x = torch.transpose(x, -1, -2)

        x = self.conv_1(x)
        # x = self.pooling(x)
        x = self.conv_2(x)
        # x = self.pooling(x)
        x = self.conv_3(x)
        # x = self.pooling(x)
        x = self.conv_4(x)
        # x = self.pooling(x)
        x = self.conv_5(x)
        x = self.pooling(x)

        x = x.view(x.size(0), -1)
        x = self.fc_2(x)

        return x


class AvastNet(nn.Module):
    """
    pytorch implementation of AvastNet, which is malware detection based on raw bytes, and from
    ICLR'2018 "DEEP CONVOLUTIONAL MALWARE CLASSIFIERS CAN LEARN FROM RAW EXECUTABLES AND LABELS ONLY"
    """
    def __init__(self, vocab_size=256):
        super(AvastNet, self).__init__()

        self.embed = nn.Embedding(vocab_size,8,padding_idx=0)  # [0,255]--> 256, out_dim=8, index=0 is not contribute to gradients, initiated with 0s
        self.conv1 = nn.Conv1d(8,48,32,stride=4,bias=True)
        self.conv2 = nn.Conv1d(48,96,32,stride=4,bias=True)
        self.maxpool = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(96,128,16,stride=8,bias=True)
        self.conv4 = nn.Conv1d(128,192,16,stride=8,bias=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)   # equivalent to global average pooling

        self.fc1 = nn.Linear(192,192)
        self.fc2 = nn.Linear(192,160)
        self.fc3 = nn.Linear(160,128)

        self.out = nn.Linear(128,2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.embed(x.long())
        x = torch.transpose(x, -1, -2)  # Channel first
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x).squeeze()
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out(x)
        # x = self.sigmoid(x)

        return x


class AvastNet_freezeEmbed(nn.Module):
    """
    freeze embed layer, embed input before sending to avastNet.
    pytorch implementation of AvastNet, which is malware detection based on raw bytes, and from
    ICLR'2018 "DEEP CONVOLUTIONAL MALWARE CLASSIFIERS CAN LEARN FROM RAW EXECUTABLES AND LABELS ONLY"
    """
    def __init__(self, vocab_size=256):
        super(AvastNet_freezeEmbed, self).__init__()

        self.embed = nn.Embedding(vocab_size,8,padding_idx=0)  # [0,255]--> 256, out_dim=8, index=0 is not contribute to gradients, initiated with 0s
        self.conv1 = nn.Conv1d(8,48,32,stride=4,bias=True)
        self.conv2 = nn.Conv1d(48,96,32,stride=4,bias=True)
        self.maxpool = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(96,128,16,stride=8,bias=True)
        self.conv4 = nn.Conv1d(128,192,16,stride=8,bias=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)   # equivalent to global average pooling

        self.fc1 = nn.Linear(192,192)
        self.fc2 = nn.Linear(192,160)
        self.fc3 = nn.Linear(160,128)

        self.out = nn.Linear(128,2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        # x = self.embed(x.long())
        x = torch.transpose(x, -1, -2)  # Channel first
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x).squeeze()
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out(x)
        # x = self.sigmoid(x)

        return x



## imageNet to classify malware based image data
class ImageNet(nn.Module):
    def __init__(self):
        # super(ImageNet, self).__init__()
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


"""
borrow from "https://github.com/Mayurji/Image-Classification-PyTorch/blob/main/AlexNet.py"

Before AlexNet, SIFT(scale-invariant feature transform), SURF or HOG were some of the hand tuned feature extractors for Computer Vision.
In AlexNet, Interestingly in the lowest layers of the network, the model learned feature extractors that resembled some traditional filters.
Higher layers in the network might build upon these representations to represent larger structures, like eyes, noses, blades of grass, and so on.
Even higher layers might represent whole objects like people, airplanes, dogs, or frisbees. Ultimately, the final hidden state learns a compact
representation of the image that summarizes its contents such that data belonging to different categories can be easily separated.
Challenges perceived before AlexNet:
Computational Power:
Due to the limited memory in early GPUs, the original AlexNet used a dual data stream design, so that each of their two GPUs could be responsible
for storing and computing only its half of the model. Fortunately, GPU memory is comparatively abundant now, so we rarely need to break up models
across GPUs these days.
Data Availability:
ImageNet was released during this period by researchers under Fei-Fei Li with 1 million images, 1000 images per class with total of 1000 class.
Note:
Instead of using ImageNet, I am using MNIST and resizing the image to 224 x 224 dimension to make it justify with the AlexNet architecture.
"""

class AlexNet(nn.Module):
    def __init__(self, input_channel, n_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            # transforming (bsize x 1 x 224 x 224) to (bsize x 96 x 54 x 54)
            # From floor((n_h - k_s + p + s)/s), floor((224 - 11 + 3 + 4) / 4) => floor(219/4) => floor(55.5) => 55
            nn.Conv2d(input_channel, 96, kernel_size=11, stride=4, padding=3),  # (batch_size * 96 * 55 * 55)
            nn.ReLU(inplace=True),  # (batch_size * 96 * 55 * 55)
            # nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2))  # (batch_size * 96 * 27 * 27)
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # (batch_size * 256 * 27 * 27)
            nn.ReLU(inplace=True),
            # nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2))  # (batch_size * 256 * 13 * 13)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # (batch_size * 384 * 13 * 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # (batch_size * 384 * 13 * 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # (batch_size * 256 * 13 * 13)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (batch_size * 256 * 6 * 6)
            nn.Flatten())  # (batch_size * 9216)


        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),  # (batch_size * 4096)  #for image resolution 224
            # nn.Linear(20736, 4096),  # (batch_size * 4096)   # for image resolution 320
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),  # (batch_size * 4096)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, n_classes))  # (batch_size * 10)

        self.conv1.apply(self.init_weights)
        self.conv2.apply(self.init_weights)
        self.conv3.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fc(out)

        return out


"""
borrow from "https://github.com/Mayurji/Image-Classification-PyTorch/blob/main/AlexNet.py"

Why ResNet?
To understand the network as we add more layers, does it becomes more expressive of the
task in hand or otherwise.
Key idea of ResNet is adding more layers which acts as a Identity function, i.e. if our
underlying mapping function which the network is trying to learn is F(x) = x, then instead
of trying to learn F(x) with Conv layers between them, we can directly add an skip connection
to tend the weight and biases of F(x) to zero. This is part of the explanation from D2L.
Adding new layer led to ResNet Block in the ResNet Architecture.
In ResNet block, in addition to typical Conv layers the authors introduce a parallel identity 
mapping skipping the conv layers to directly connect the input with output of conv layers.
A such connection is termed as Skip Connection or Residual connection.
Things to note while adding the skip connection to output conv block is the dimensions.Important
to note, as mentioned earlier in NIN network, we can use 1x1 Conv to increase and decrease the 
dimension.
Below is a ResNet18 architecture:
There are 4 convolutional layers in each module (excluding the 1×1 convolutional layer). 
Together with the first 7×7 convolutional layer and the final fully-connected layer, there are 
18 layers in total. Therefore, this model is a ResNet-18.
"""
import torch.nn as nn
from torch.nn import functional as F


class Residual(nn.Module):
    """
    ResNet 18
    """
    def __init__(self, in_channel, out_channel, use_1x1Conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        if use_1x1Conv:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, X):
        out = F.relu(self.bn1(self.conv1(X)))
        out = self.bn2(self.conv2(out))
        if self.conv3:
            X = self.conv3(X)
        out += X
        return F.relu(out)


def residualBlock(in_channel, out_channel, num_residuals, first_block=False):
    blks = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blks.append(Residual(in_channel, out_channel, use_1x1Conv=True,
                                 strides=2))
        else:
            blks.append(Residual(out_channel, out_channel))

    return blks


class ResNet18(nn.Module):
    def __init__(self, input_channel, n_classes):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*residualBlock(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*residualBlock(64, 128, 2))
        self.b4 = nn.Sequential(*residualBlock(128, 256, 2))
        self.b5 = nn.Sequential(*residualBlock(256, 512, 2))
        self.finalLayer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), nn.Linear(512, n_classes))

        self.b1.apply(self.init_weights)
        self.b2.apply(self.init_weights)
        self.b3.apply(self.init_weights)
        self.b4.apply(self.init_weights)
        self.b5.apply(self.init_weights)
        self.finalLayer.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Conv2d:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out')
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, std=1e-3)
        if type(layer) == nn.BatchNorm2d:
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)

    def forward(self, X):
        out = self.b1(X)
        out = self.b2(out)
        out = self.b3(out)
        out = self.b4(out)
        out = self.b5(out)
        out = self.finalLayer(out)

        return out


"""
borrow from "https://github.com/Mayurji/Image-Classification-PyTorch/blob/main/AlexNet.py"

Xception
The network uses a modified version of Depthwise Seperable Convolution. It combines
ideas from MobileNetV1 like depthwise seperable conv and from InceptionV3, the order 
of the layers like conv1x1 and then spatial kernels.
In modified Depthwise Seperable Convolution network, the order of operation is changed
by keeping Conv1x1 and then the spatial convolutional kernel. And the other difference
is the absence of Non-Linear activation function. And with inclusion of residual 
connections impacts the performs of Xception widely.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SeparableConv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.dwc = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, kernel_size, stride, padding, dilation, groups=input_channel,
                      bias=bias),
            nn.Conv2d(input_channel, output_channel, 1, 1, 0, 1, 1, bias=bias)
        )

    def forward(self, X):
        return self.dwc(X)


class Block(nn.Module):
    def __init__(self, input_channel, out_channel, reps, strides=1, relu=True, grow_first=True):
        super().__init__()
        if out_channel != input_channel or strides != 1:
            self.skipConnection = nn.Sequential(
                nn.Conv2d(input_channel, out_channel, 1, stride=strides, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.skipConnection = None
        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = input_channel
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv(input_channel, out_channel, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_channel))
            filters = out_channel

        for _ in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv(input_channel, out_channel, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_channel))

        if not relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self, input):
        X = self.rep(input)

        if self.skipConnection:
            skip = self.skipConnection(input)
        else:
            skip = input

        X += skip
        return X


class Xception(nn.Module):
    def __init__(self, input_channel, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.relu = nn.ReLU(inplace=True)

        self.initBlock = nn.Sequential(
            nn.Conv2d(input_channel, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.block1 = Block(64, 128, 2, 2, relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, relu=True, grow_first=False)

        self.conv3 = SeparableConv(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, self.n_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.initBlock(x)
        x = self.block1(x)

        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


"""
ResNet34: https://github.com/steven0129/pytorch-ResNet34/blob/master/main.py
from "
"""
class ResidualBlock(nn.Module):
    '''
    實現子Module: Residual Block
    '''

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        resisdual = x if self.right is None else self.right(x)
        out += resisdual
        return F.relu(out)


class ResNet34(nn.Module):
    '''
    實現主Module: ResNet34
    ResNet34包含多個layer，每個layer又包含多個Residual block
    用子Module來實現Residual Block，用make_layer函數來實現layer
    '''

    def __init__(self, input_channel=1,num_classes=2):
        super(ResNet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(input_channel, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        # 分類的Layer，分別有3, 4, 6個Residual Block
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)

        # 分類用的Fully Connection
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        '''
        構建Layer，包含多個Residual Block
        '''
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)
