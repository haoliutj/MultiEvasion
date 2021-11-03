import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from secml_malware.models.basee2e import End2EndModel


class MalConv(nn.Module):
    "keep Embed layer for traning"
    def __init__(self, max_input_size=102400, window_size=500):
        super(MalConv, self).__init__()

        self.embed = nn.Embedding(257, 8, padding_idx=0)    # [1,256], 0,in total 257

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
        #x = self.sigmoid(x)

        return x


class MalConv_freezeEmbed(nn.Module):
    "produce adve_x: freeze the embed layer, to produce aadversarial example"

    def __init__(self, max_input_size=102400, window_size=500):
        super(MalConv_freezeEmbed, self).__init__()

        self.embed = nn.Embedding(257, 8, padding_idx=0)  # [1,256], 0,in total 257

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



# class MalConv_allin(nn.Module):
#     "train detector: eat all exe file; unfreeze embed layer"
#
#     # trained to minimize cross-entropy loss
#     # criterion = nn.CrossEntropyLoss()
#     def __init__(self, out_size=2, channels=128, window_size=512, embd_size=8):
#         super(MalConv_allin, self).__init__()
#         self.embd = nn.Embedding(257, embd_size, padding_idx=0)
#
#         self.window_size = window_size
#
#         self.conv_1 = nn.Conv1d(embd_size, channels, window_size, stride=window_size, bias=True)
#         self.conv_2 = nn.Conv1d(embd_size, channels, window_size, stride=window_size, bias=True)
#
#         self.pooling = nn.AdaptiveMaxPool1d(1)
#
#         self.fc_1 = nn.Linear(channels, channels)
#         self.fc_2 = nn.Linear(channels, out_size)
#
#     def forward(self, x):
#         x = self.embd(x.long())
#         x = torch.transpose(x, -1, -2)
#
#         cnn_value = self.conv_1(x)
#         gating_weight = torch.sigmoid(self.conv_2(x))
#
#         x = cnn_value * gating_weight
#
#         x = self.pooling(x)
#
#         # Flatten
#         x = x.view(x.size(0), -1)
#
#         x = F.relu(self.fc_1(x))
#         x = self.fc_2(x)
#
#         return x











# class MalConv_1(nn.Module):
#     # trained to minimize cross-entropy loss
#     # criterion = nn.CrossEntropyLoss()
#     def __init__(self, out_size=2, channels=128, window_size=512, embd_size=8):
#         super(MalConv_1, self).__init__()
#         self.embd = nn.Embedding(257, embd_size, padding_idx=0)
#
#         self.window_size = window_size
#
#         self.conv_1 = nn.Conv1d(embd_size, channels, window_size, stride=window_size, bias=True)
#         self.conv_2 = nn.Conv1d(embd_size, channels, window_size, stride=window_size, bias=True)
#
#         self.pooling = nn.AdaptiveMaxPool1d(1)
#
#         self.fc_1 = nn.Linear(channels, channels)
#         self.fc_2 = nn.Linear(channels, out_size)
#
#     def forward(self, x):
#
#         x = torch.transpose(x, -1, -2)
#
#         cnn_value = self.conv_1(x)
#         gating_weight = torch.sigmoid(self.conv_2(x))
#
#         x = cnn_value * gating_weight
#
#         x = self.pooling(x)
#
#         # Flatten
#         x = x.view(x.size(0), -1)
#
#         x = F.relu(self.fc_1(x))
#         x = self.fc_2(x)
#
#         return x
#
#
# class MalConv_2(nn.Module):
#     # trained to minimize cross-entropy loss
#     # criterion = nn.CrossEntropyLoss()
#     def __init__(self, out_size=2, channels=128, window_size=512, embd_size=8):
#         super(MalConv_2, self).__init__()
#         self.embd = nn.Embedding(257, embd_size, padding_idx=0)
#
#         self.window_size = window_size
#
#         self.conv_1 = nn.Conv1d(embd_size, channels, window_size, stride=window_size, bias=True)
#         self.conv_2 = nn.Conv1d(embd_size, channels, window_size, stride=window_size, bias=True)
#
#         self.pooling = nn.AdaptiveMaxPool1d(1)
#
#         self.fc_1 = nn.Linear(channels, channels)
#         self.fc_2 = nn.Linear(channels, out_size)
#
#     def forward(self, x):
#         x = self.embd(x.long())
#         x = torch.transpose(x, -1, -2)
#
#         cnn_value = self.conv_1(x)
#         gating_weight = torch.sigmoid(self.conv_2(x))
#
#         x = cnn_value * gating_weight
#
#         x = self.pooling(x)
#
#         # Flatten
#         x = x.view(x.size(0), -1)
#
#         x = F.relu(self.fc_1(x))
#         x = self.fc_2(x)
#
#         return x



# class FireEyeCNN(nn.Module):
#     def __init__(self, input_length=2000000, window_size=32):
#         super(FireEyeCNN, self).__init__()
#
#         self.embed = nn.Embedding(257, 8, padding_idx=0)
#
#         self.conv_1 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
#         self.conv_2 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
#         self.conv_3 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
#         self.conv_4 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
#         self.conv_5 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
#
#
#         self.pooling = nn.MaxPool1d(int(input_length / window_size))
#
#         self.fc_1 = nn.Linear(128, 128)
#         self.fc_2 = nn.Linear(128, 2)
#
#         self.sigmoid = nn.Sigmoid() # binary classification
#         # self.softmax = nn.Softmax()   # multi-classfication
#
#     def forward(self, x):
#         x = self.embed(x.long())
#         # Channel first
#         x = torch.transpose(x, -1, -2)
#
#         cnn_value = self.conv_1(x.narrow(-2, 0, 4))
#         cnn_value = self.conv_2(x.narrow(-2, 0, 4))
#         cnn_value = self.conv_3(x.narrow(-2, 0, 4))
#         cnn_value = self.conv_4(x.narrow(-2, 0, 4))
#
#         gating_weight = self.sigmoid(self.conv_5(x.narrow(-2, 4, 4)))
#
#         x = cnn_value * gating_weight
#         x = self.pooling(x)
#
#         x = x.view(-1, 128)
#         x = self.fc_1(x)
#         x = self.fc_2(x)
#         # x = self.sigmoid(x)
#
#         return x


class FireEye(nn.Module):
    """
    DNN model from "Activation Analysis of a Byte-Based Deep Neural Network for Malware Classification" FireEye Inc.
    keep embed layer for training
    """
    def __init__(self, input_length=102400, window_size=512):
        super(FireEye, self).__init__()

        self.embed = nn.Embedding(257, 10, padding_idx=0)

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

    def __init__(self, input_length=102400, window_size=512):
        super(FireEye_freezeEmbed, self).__init__()

        self.embed = nn.Embedding(257, 10, padding_idx=0)

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



# class cnn(nn.Module):
#     "works for the data that already normalized"
#
#     def __init__(self,params):
#         self.params = params
#         super(cnn, self).__init__()
#
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(in_channels=self.params['conv1_input_channel'], out_channels=self.params['conv1_output_channel'],
#                       kernel_size=self.params['kernel_size1'], stride=self.params['stride1'], padding=self.params['padding1']),
#             nn.ReLU(),
#             nn.Dropout(self.params['drop_rate1']),
#             nn.MaxPool1d(kernel_size=self.params['pool1'])
#         )
#
#         self.conv2 = nn.Sequential(
#             nn.Conv1d(in_channels=self.params['conv2_input_channel'], out_channels=self.params['conv2_output_channel'],
#                       kernel_size=self.params['kernel_size2'], stride=self.params['stride2'],
#                       padding=self.params['padding2']),
#             nn.ReLU(),
#             nn.Dropout(self.params['drop_rate2']),
#             nn.MaxPool1d(kernel_size=self.params['pool2'])
#         )
#
#         self.conv3 = nn.Sequential(
#             nn.Conv1d(in_channels=self.params['conv3_input_channel'], out_channels=self.params['conv3_output_channel'],
#                       kernel_size=self.params['kernel_size3'], stride=self.params['stride3'],
#                       padding=self.params['padding3']),
#             nn.ReLU(),
#             nn.Dropout(self.params['drop_rate3']),
#             nn.MaxPool1d(kernel_size=self.params['pool3'])
#         )
#
#         self.conv4 = nn.Sequential(
#             nn.Conv1d(in_channels=self.params['conv4_input_channel'], out_channels=self.params['conv4_output_channel'],
#                       kernel_size=self.params['kernel_size4'], stride=self.params['stride4'],
#                       padding=self.params['padding4']),
#             nn.ReLU(),
#             nn.Dropout(self.params['drop_rate4']),
#             nn.MaxPool1d(kernel_size=self.params['pool4'])
#         )
#
#         #add flatten layer, the output dimension of flatten is the input of FC num_layers
#         # self.flatten = nn.Flatten()
#         # self.out_param1 = self.flatten.size[1]
#
#         self.out_param1 = math.ceil(math.ceil(math.ceil(math.ceil(self.params['input_size']/self.params['pool1'])/self.params['pool2'])/self.params['pool3'])/self.params['pool4'])
#
#         self.out = nn.Linear(self.params['conv4_output_channel']*self.out_param1,self.params['num_classes'])
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = x.view(x.size(0), -1)
#         logits = self.out(x)
#         return logits
