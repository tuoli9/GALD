import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import pandas as pd
from tqdm import tqdm
from PIL import Image
import csv

_weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

class Res101(nn.Module):

    
    def __init__(self, weight_file):
        super(Res101, self).__init__()
        global _weights_dict
        _weights_dict = load_weights(weight_file)

        self.resnet_v2_101_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/conv1/Conv2D', in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), groups=1, bias=True)
        self.resnet_v2_101_block1_unit_1_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block1/unit_1/bottleneck_v2/preact/FusedBatchNorm', num_features=64, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block1_unit_1_bottleneck_v2_shortcut_Conv2D = self.__conv(2, name='resnet_v2_101/block1/unit_1/bottleneck_v2/shortcut/Conv2D', in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block1_unit_1_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block1/unit_1/bottleneck_v2/conv1/Conv2D', in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block1_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block1/unit_1/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=64, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block1_unit_1_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block1/unit_1/bottleneck_v2/conv2/Conv2D', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block1_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block1/unit_1/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=64, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block1_unit_1_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block1/unit_1/bottleneck_v2/conv3/Conv2D', in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block1_unit_2_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block1/unit_2/bottleneck_v2/preact/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block1_unit_2_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block1/unit_2/bottleneck_v2/conv1/Conv2D', in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block1_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block1/unit_2/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=64, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block1_unit_2_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block1/unit_2/bottleneck_v2/conv2/Conv2D', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block1_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block1/unit_2/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=64, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block1_unit_2_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block1/unit_2/bottleneck_v2/conv3/Conv2D', in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block1_unit_3_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block1/unit_3/bottleneck_v2/preact/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block1_unit_3_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block1/unit_3/bottleneck_v2/conv1/Conv2D', in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block1_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block1/unit_3/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=64, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block1_unit_3_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block1/unit_3/bottleneck_v2/conv2/Conv2D', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=None)
        self.resnet_v2_101_block1_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block1/unit_3/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=64, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block1_unit_3_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block1/unit_3/bottleneck_v2/conv3/Conv2D', in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block2_unit_1_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block2/unit_1/bottleneck_v2/preact/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block2_unit_1_bottleneck_v2_shortcut_Conv2D = self.__conv(2, name='resnet_v2_101/block2/unit_1/bottleneck_v2/shortcut/Conv2D', in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block2_unit_1_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block2/unit_1/bottleneck_v2/conv1/Conv2D', in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block2_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block2/unit_1/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=128, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block2_unit_1_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block2/unit_1/bottleneck_v2/conv2/Conv2D', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block2_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block2/unit_1/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=128, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block2_unit_1_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block2/unit_1/bottleneck_v2/conv3/Conv2D', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block2_unit_2_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block2/unit_2/bottleneck_v2/preact/FusedBatchNorm', num_features=512, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block2_unit_2_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block2/unit_2/bottleneck_v2/conv1/Conv2D', in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block2_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block2/unit_2/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=128, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block2_unit_2_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block2/unit_2/bottleneck_v2/conv2/Conv2D', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block2_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block2/unit_2/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=128, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block2_unit_2_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block2/unit_2/bottleneck_v2/conv3/Conv2D', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block2_unit_3_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block2/unit_3/bottleneck_v2/preact/FusedBatchNorm', num_features=512, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block2_unit_3_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block2/unit_3/bottleneck_v2/conv1/Conv2D', in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block2_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block2/unit_3/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=128, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block2_unit_3_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block2/unit_3/bottleneck_v2/conv2/Conv2D', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block2_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block2/unit_3/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=128, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block2_unit_3_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block2/unit_3/bottleneck_v2/conv3/Conv2D', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block2_unit_4_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block2/unit_4/bottleneck_v2/preact/FusedBatchNorm', num_features=512, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block2_unit_4_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block2/unit_4/bottleneck_v2/conv1/Conv2D', in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block2_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block2/unit_4/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=128, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block2_unit_4_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block2/unit_4/bottleneck_v2/conv2/Conv2D', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=None)
        self.resnet_v2_101_block2_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block2/unit_4/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=128, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block2_unit_4_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block2/unit_4/bottleneck_v2/conv3/Conv2D', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_1_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_1/bottleneck_v2/preact/FusedBatchNorm', num_features=512, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_1_bottleneck_v2_shortcut_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_1/bottleneck_v2/shortcut/Conv2D', in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_1_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_1/bottleneck_v2/conv1/Conv2D', in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_1/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_1_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_1/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_1/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_1_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_1/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_2_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_2/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_2_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_2/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_2/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_2_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_2/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_2/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_2_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_2/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_3_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_3/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_3_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_3/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_3/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_3_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_3/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_3/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_3_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_3/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_4_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_4/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_4_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_4/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_4/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_4_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_4/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_4/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_4_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_4/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_5_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_5/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_5_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_5/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_5_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_5/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_5_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_5/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_5_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_5/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_5_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_5/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_6_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_6/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_6_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_6/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_6_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_6/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_6_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_6/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_6_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_6/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_6_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_6/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_7_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_7/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_7_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_7/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_7_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_7/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_7_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_7/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_7_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_7/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_7_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_7/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_8_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_8/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_8_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_8/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_8_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_8/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_8_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_8/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_8_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_8/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_8_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_8/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_9_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_9/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_9_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_9/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_9_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_9/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_9_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_9/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_9_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_9/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_9_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_9/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_10_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_10/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_10_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_10/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_10_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_10/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_10_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_10/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_10_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_10/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_10_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_10/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_11_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_11/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_11_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_11/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_11_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_11/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_11_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_11/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_11_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_11/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_11_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_11/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_12_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_12/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_12_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_12/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_12_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_12/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_12_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_12/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_12_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_12/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_12_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_12/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_13_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_13/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_13_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_13/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_13_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_13/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_13_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_13/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_13_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_13/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_13_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_13/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_14_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_14/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_14_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_14/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_14_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_14/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_14_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_14/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_14_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_14/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_14_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_14/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_15_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_15/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_15_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_15/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_15_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_15/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_15_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_15/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_15_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_15/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_15_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_15/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_16_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_16/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_16_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_16/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_16_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_16/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_16_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_16/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_16_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_16/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_16_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_16/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_17_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_17/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_17_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_17/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_17_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_17/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_17_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_17/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_17_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_17/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_17_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_17/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_18_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_18/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_18_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_18/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_18_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_18/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_18_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_18/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_18_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_18/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_18_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_18/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_19_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_19/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_19_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_19/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_19_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_19/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_19_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_19/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_19_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_19/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_19_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_19/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_20_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_20/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_20_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_20/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_20_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_20/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_20_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_20/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_20_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_20/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_20_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_20/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_21_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_21/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_21_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_21/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_21_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_21/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_21_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_21/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_21_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_21/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_21_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_21/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_22_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_22/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_22_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_22/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_22_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_22/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_22_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_22/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_22_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_22/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_22_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_22/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block3_unit_23_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_23/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_23_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_23/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_23_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_23/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_23_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_23/bottleneck_v2/conv2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=None)
        self.resnet_v2_101_block3_unit_23_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block3/unit_23/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=256, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block3_unit_23_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block3/unit_23/bottleneck_v2/conv3/Conv2D', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block4_unit_1_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block4/unit_1/bottleneck_v2/preact/FusedBatchNorm', num_features=1024, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block4_unit_1_bottleneck_v2_shortcut_Conv2D = self.__conv(2, name='resnet_v2_101/block4/unit_1/bottleneck_v2/shortcut/Conv2D', in_channels=1024, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block4_unit_1_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block4/unit_1/bottleneck_v2/conv1/Conv2D', in_channels=1024, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block4_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block4/unit_1/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=512, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block4_unit_1_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block4/unit_1/bottleneck_v2/conv2/Conv2D', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block4_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block4/unit_1/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=512, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block4_unit_1_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block4/unit_1/bottleneck_v2/conv3/Conv2D', in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block4_unit_2_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block4/unit_2/bottleneck_v2/preact/FusedBatchNorm', num_features=2048, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block4_unit_2_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block4/unit_2/bottleneck_v2/conv1/Conv2D', in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block4_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block4/unit_2/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=512, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block4_unit_2_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block4/unit_2/bottleneck_v2/conv2/Conv2D', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block4_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block4/unit_2/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=512, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block4_unit_2_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block4/unit_2/bottleneck_v2/conv3/Conv2D', in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_block4_unit_3_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block4/unit_3/bottleneck_v2/preact/FusedBatchNorm', num_features=2048, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block4_unit_3_bottleneck_v2_conv1_Conv2D = self.__conv(2, name='resnet_v2_101/block4/unit_3/bottleneck_v2/conv1/Conv2D', in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block4_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block4/unit_3/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm', num_features=512, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block4_unit_3_bottleneck_v2_conv2_Conv2D = self.__conv(2, name='resnet_v2_101/block4/unit_3/bottleneck_v2/conv2/Conv2D', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.resnet_v2_101_block4_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/block4/unit_3/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm', num_features=512, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_block4_unit_3_bottleneck_v2_conv3_Conv2D = self.__conv(2, name='resnet_v2_101/block4/unit_3/bottleneck_v2/conv3/Conv2D', in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.resnet_v2_101_postnorm_FusedBatchNorm = self.__batch_normalization(2, 'resnet_v2_101/postnorm/FusedBatchNorm', num_features=2048, eps=1.0009999641624745e-05, momentum=0.0)
        self.resnet_v2_101_logits_Conv2D = self.__conv(2, name='resnet_v2_101/logits/Conv2D', in_channels=2048, out_channels=1001, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)

    def forward(self, x):
        resnet_v2_101_Pad = F.pad(x, (3, 3, 3, 3), mode = 'constant', value = 0)
        resnet_v2_101_conv1_Conv2D = self.resnet_v2_101_conv1_Conv2D(resnet_v2_101_Pad)
        resnet_v2_101_pool1_MaxPool_pad = F.pad(resnet_v2_101_conv1_Conv2D, (0, 1, 0, 1), value=float('-inf'))
        resnet_v2_101_pool1_MaxPool, resnet_v2_101_pool1_MaxPool_idx = F.max_pool2d(resnet_v2_101_pool1_MaxPool_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        resnet_v2_101_block1_unit_1_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block1_unit_1_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_pool1_MaxPool)
        resnet_v2_101_block1_unit_1_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block1_unit_1_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block1_unit_1_bottleneck_v2_shortcut_Conv2D = self.resnet_v2_101_block1_unit_1_bottleneck_v2_shortcut_Conv2D(resnet_v2_101_block1_unit_1_bottleneck_v2_preact_Relu)
        resnet_v2_101_block1_unit_1_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block1_unit_1_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block1_unit_1_bottleneck_v2_preact_Relu)
        resnet_v2_101_block1_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block1_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block1_unit_1_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block1_unit_1_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block1_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block1_unit_1_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block1_unit_1_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block1_unit_1_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block1_unit_1_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block1_unit_1_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block1_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block1_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block1_unit_1_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block1_unit_1_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block1_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block1_unit_1_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block1_unit_1_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block1_unit_1_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block1_unit_1_bottleneck_v2_add = resnet_v2_101_block1_unit_1_bottleneck_v2_shortcut_Conv2D + resnet_v2_101_block1_unit_1_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block1_unit_2_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block1_unit_2_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block1_unit_1_bottleneck_v2_add)
        resnet_v2_101_block1_unit_2_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block1_unit_2_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block1_unit_2_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block1_unit_2_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block1_unit_2_bottleneck_v2_preact_Relu)
        resnet_v2_101_block1_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block1_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block1_unit_2_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block1_unit_2_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block1_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block1_unit_2_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block1_unit_2_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block1_unit_2_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block1_unit_2_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block1_unit_2_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block1_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block1_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block1_unit_2_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block1_unit_2_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block1_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block1_unit_2_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block1_unit_2_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block1_unit_2_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block1_unit_2_bottleneck_v2_add = resnet_v2_101_block1_unit_1_bottleneck_v2_add + resnet_v2_101_block1_unit_2_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block1_unit_3_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block1_unit_3_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block1_unit_2_bottleneck_v2_add)
        resnet_v2_101_block1_unit_3_bottleneck_v2_shortcut_MaxPool, resnet_v2_101_block1_unit_3_bottleneck_v2_shortcut_MaxPool_idx = F.max_pool2d(resnet_v2_101_block1_unit_2_bottleneck_v2_add, kernel_size=(1, 1), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        resnet_v2_101_block1_unit_3_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block1_unit_3_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block1_unit_3_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block1_unit_3_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block1_unit_3_bottleneck_v2_preact_Relu)
        resnet_v2_101_block1_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block1_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block1_unit_3_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block1_unit_3_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block1_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block1_unit_3_bottleneck_v2_Pad = F.pad(resnet_v2_101_block1_unit_3_bottleneck_v2_conv1_Relu, (1, 1, 1, 1), mode = 'constant', value = 0)
        resnet_v2_101_block1_unit_3_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block1_unit_3_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block1_unit_3_bottleneck_v2_Pad)
        resnet_v2_101_block1_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block1_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block1_unit_3_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block1_unit_3_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block1_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block1_unit_3_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block1_unit_3_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block1_unit_3_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block1_unit_3_bottleneck_v2_add = resnet_v2_101_block1_unit_3_bottleneck_v2_shortcut_MaxPool + resnet_v2_101_block1_unit_3_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block2_unit_1_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block2_unit_1_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block1_unit_3_bottleneck_v2_add)
        resnet_v2_101_block2_unit_1_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block2_unit_1_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block2_unit_1_bottleneck_v2_shortcut_Conv2D = self.resnet_v2_101_block2_unit_1_bottleneck_v2_shortcut_Conv2D(resnet_v2_101_block2_unit_1_bottleneck_v2_preact_Relu)
        resnet_v2_101_block2_unit_1_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block2_unit_1_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block2_unit_1_bottleneck_v2_preact_Relu)
        resnet_v2_101_block2_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block2_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block2_unit_1_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block2_unit_1_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block2_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block2_unit_1_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block2_unit_1_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block2_unit_1_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block2_unit_1_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block2_unit_1_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block2_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block2_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block2_unit_1_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block2_unit_1_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block2_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block2_unit_1_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block2_unit_1_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block2_unit_1_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block2_unit_1_bottleneck_v2_add = resnet_v2_101_block2_unit_1_bottleneck_v2_shortcut_Conv2D + resnet_v2_101_block2_unit_1_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block2_unit_2_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block2_unit_2_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block2_unit_1_bottleneck_v2_add)
        resnet_v2_101_block2_unit_2_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block2_unit_2_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block2_unit_2_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block2_unit_2_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block2_unit_2_bottleneck_v2_preact_Relu)
        resnet_v2_101_block2_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block2_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block2_unit_2_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block2_unit_2_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block2_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block2_unit_2_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block2_unit_2_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block2_unit_2_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block2_unit_2_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block2_unit_2_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block2_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block2_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block2_unit_2_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block2_unit_2_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block2_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block2_unit_2_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block2_unit_2_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block2_unit_2_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block2_unit_2_bottleneck_v2_add = resnet_v2_101_block2_unit_1_bottleneck_v2_add + resnet_v2_101_block2_unit_2_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block2_unit_3_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block2_unit_3_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block2_unit_2_bottleneck_v2_add)
        resnet_v2_101_block2_unit_3_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block2_unit_3_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block2_unit_3_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block2_unit_3_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block2_unit_3_bottleneck_v2_preact_Relu)
        resnet_v2_101_block2_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block2_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block2_unit_3_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block2_unit_3_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block2_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block2_unit_3_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block2_unit_3_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block2_unit_3_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block2_unit_3_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block2_unit_3_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block2_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block2_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block2_unit_3_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block2_unit_3_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block2_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block2_unit_3_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block2_unit_3_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block2_unit_3_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block2_unit_3_bottleneck_v2_add = resnet_v2_101_block2_unit_2_bottleneck_v2_add + resnet_v2_101_block2_unit_3_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block2_unit_4_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block2_unit_4_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block2_unit_3_bottleneck_v2_add)
        resnet_v2_101_block2_unit_4_bottleneck_v2_shortcut_MaxPool, resnet_v2_101_block2_unit_4_bottleneck_v2_shortcut_MaxPool_idx = F.max_pool2d(resnet_v2_101_block2_unit_3_bottleneck_v2_add, kernel_size=(1, 1), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        resnet_v2_101_block2_unit_4_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block2_unit_4_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block2_unit_4_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block2_unit_4_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block2_unit_4_bottleneck_v2_preact_Relu)
        resnet_v2_101_block2_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block2_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block2_unit_4_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block2_unit_4_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block2_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block2_unit_4_bottleneck_v2_Pad = F.pad(resnet_v2_101_block2_unit_4_bottleneck_v2_conv1_Relu, (1, 1, 1, 1), mode = 'constant', value = 0)
        resnet_v2_101_block2_unit_4_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block2_unit_4_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block2_unit_4_bottleneck_v2_Pad)
        resnet_v2_101_block2_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block2_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block2_unit_4_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block2_unit_4_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block2_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block2_unit_4_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block2_unit_4_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block2_unit_4_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block2_unit_4_bottleneck_v2_add = resnet_v2_101_block2_unit_4_bottleneck_v2_shortcut_MaxPool + resnet_v2_101_block2_unit_4_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_1_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_1_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block2_unit_4_bottleneck_v2_add)
        resnet_v2_101_block3_unit_1_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_1_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_1_bottleneck_v2_shortcut_Conv2D = self.resnet_v2_101_block3_unit_1_bottleneck_v2_shortcut_Conv2D(resnet_v2_101_block3_unit_1_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_1_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_1_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_1_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_1_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_1_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_1_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_1_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_1_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_1_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_1_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_1_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_1_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_1_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_1_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_1_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_1_bottleneck_v2_add = resnet_v2_101_block3_unit_1_bottleneck_v2_shortcut_Conv2D + resnet_v2_101_block3_unit_1_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_2_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_2_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_1_bottleneck_v2_add)
        resnet_v2_101_block3_unit_2_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_2_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_2_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_2_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_2_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_2_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_2_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_2_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_2_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_2_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_2_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_2_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_2_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_2_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_2_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_2_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_2_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_2_bottleneck_v2_add = resnet_v2_101_block3_unit_1_bottleneck_v2_add + resnet_v2_101_block3_unit_2_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_3_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_3_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_2_bottleneck_v2_add)
        resnet_v2_101_block3_unit_3_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_3_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_3_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_3_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_3_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_3_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_3_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_3_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_3_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_3_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_3_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_3_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_3_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_3_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_3_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_3_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_3_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_3_bottleneck_v2_add = resnet_v2_101_block3_unit_2_bottleneck_v2_add + resnet_v2_101_block3_unit_3_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_4_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_4_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_3_bottleneck_v2_add)
        resnet_v2_101_block3_unit_4_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_4_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_4_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_4_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_4_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_4_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_4_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_4_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_4_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_4_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_4_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_4_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_4_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_4_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_4_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_4_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_4_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_4_bottleneck_v2_add = resnet_v2_101_block3_unit_3_bottleneck_v2_add + resnet_v2_101_block3_unit_4_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_5_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_5_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_4_bottleneck_v2_add)
        resnet_v2_101_block3_unit_5_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_5_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_5_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_5_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_5_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_5_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_5_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_5_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_5_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_5_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_5_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_5_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_5_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_5_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_5_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_5_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_5_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_5_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_5_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_5_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_5_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_5_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_5_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_5_bottleneck_v2_add = resnet_v2_101_block3_unit_4_bottleneck_v2_add + resnet_v2_101_block3_unit_5_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_6_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_6_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_5_bottleneck_v2_add)
        resnet_v2_101_block3_unit_6_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_6_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_6_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_6_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_6_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_6_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_6_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_6_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_6_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_6_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_6_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_6_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_6_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_6_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_6_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_6_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_6_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_6_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_6_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_6_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_6_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_6_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_6_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_6_bottleneck_v2_add = resnet_v2_101_block3_unit_5_bottleneck_v2_add + resnet_v2_101_block3_unit_6_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_7_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_7_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_6_bottleneck_v2_add)
        resnet_v2_101_block3_unit_7_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_7_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_7_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_7_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_7_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_7_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_7_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_7_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_7_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_7_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_7_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_7_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_7_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_7_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_7_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_7_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_7_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_7_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_7_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_7_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_7_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_7_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_7_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_7_bottleneck_v2_add = resnet_v2_101_block3_unit_6_bottleneck_v2_add + resnet_v2_101_block3_unit_7_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_8_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_8_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_7_bottleneck_v2_add)
        resnet_v2_101_block3_unit_8_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_8_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_8_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_8_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_8_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_8_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_8_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_8_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_8_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_8_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_8_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_8_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_8_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_8_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_8_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_8_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_8_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_8_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_8_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_8_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_8_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_8_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_8_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_8_bottleneck_v2_add = resnet_v2_101_block3_unit_7_bottleneck_v2_add + resnet_v2_101_block3_unit_8_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_9_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_9_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_8_bottleneck_v2_add)
        resnet_v2_101_block3_unit_9_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_9_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_9_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_9_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_9_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_9_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_9_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_9_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_9_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_9_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_9_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_9_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_9_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_9_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_9_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_9_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_9_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_9_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_9_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_9_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_9_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_9_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_9_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_9_bottleneck_v2_add = resnet_v2_101_block3_unit_8_bottleneck_v2_add + resnet_v2_101_block3_unit_9_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_10_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_10_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_9_bottleneck_v2_add)
        resnet_v2_101_block3_unit_10_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_10_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_10_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_10_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_10_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_10_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_10_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_10_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_10_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_10_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_10_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_10_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_10_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_10_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_10_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_10_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_10_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_10_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_10_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_10_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_10_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_10_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_10_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_10_bottleneck_v2_add = resnet_v2_101_block3_unit_9_bottleneck_v2_add + resnet_v2_101_block3_unit_10_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_11_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_11_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_10_bottleneck_v2_add)
        resnet_v2_101_block3_unit_11_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_11_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_11_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_11_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_11_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_11_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_11_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_11_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_11_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_11_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_11_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_11_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_11_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_11_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_11_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_11_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_11_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_11_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_11_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_11_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_11_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_11_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_11_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_11_bottleneck_v2_add = resnet_v2_101_block3_unit_10_bottleneck_v2_add + resnet_v2_101_block3_unit_11_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_12_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_12_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_11_bottleneck_v2_add)
        resnet_v2_101_block3_unit_12_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_12_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_12_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_12_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_12_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_12_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_12_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_12_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_12_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_12_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_12_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_12_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_12_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_12_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_12_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_12_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_12_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_12_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_12_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_12_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_12_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_12_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_12_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_12_bottleneck_v2_add = resnet_v2_101_block3_unit_11_bottleneck_v2_add + resnet_v2_101_block3_unit_12_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_13_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_13_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_12_bottleneck_v2_add)
        resnet_v2_101_block3_unit_13_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_13_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_13_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_13_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_13_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_13_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_13_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_13_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_13_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_13_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_13_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_13_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_13_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_13_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_13_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_13_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_13_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_13_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_13_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_13_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_13_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_13_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_13_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_13_bottleneck_v2_add = resnet_v2_101_block3_unit_12_bottleneck_v2_add + resnet_v2_101_block3_unit_13_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_14_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_14_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_13_bottleneck_v2_add)
        resnet_v2_101_block3_unit_14_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_14_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_14_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_14_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_14_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_14_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_14_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_14_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_14_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_14_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_14_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_14_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_14_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_14_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_14_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_14_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_14_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_14_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_14_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_14_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_14_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_14_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_14_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_14_bottleneck_v2_add = resnet_v2_101_block3_unit_13_bottleneck_v2_add + resnet_v2_101_block3_unit_14_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_15_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_15_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_14_bottleneck_v2_add)
        resnet_v2_101_block3_unit_15_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_15_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_15_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_15_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_15_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_15_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_15_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_15_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_15_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_15_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_15_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_15_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_15_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_15_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_15_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_15_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_15_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_15_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_15_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_15_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_15_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_15_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_15_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_15_bottleneck_v2_add = resnet_v2_101_block3_unit_14_bottleneck_v2_add + resnet_v2_101_block3_unit_15_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_16_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_16_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_15_bottleneck_v2_add)
        resnet_v2_101_block3_unit_16_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_16_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_16_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_16_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_16_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_16_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_16_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_16_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_16_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_16_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_16_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_16_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_16_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_16_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_16_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_16_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_16_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_16_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_16_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_16_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_16_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_16_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_16_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_16_bottleneck_v2_add = resnet_v2_101_block3_unit_15_bottleneck_v2_add + resnet_v2_101_block3_unit_16_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_17_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_17_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_16_bottleneck_v2_add)
        resnet_v2_101_block3_unit_17_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_17_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_17_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_17_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_17_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_17_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_17_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_17_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_17_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_17_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_17_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_17_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_17_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_17_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_17_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_17_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_17_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_17_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_17_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_17_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_17_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_17_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_17_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_17_bottleneck_v2_add = resnet_v2_101_block3_unit_16_bottleneck_v2_add + resnet_v2_101_block3_unit_17_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_18_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_18_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_17_bottleneck_v2_add)
        resnet_v2_101_block3_unit_18_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_18_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_18_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_18_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_18_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_18_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_18_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_18_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_18_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_18_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_18_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_18_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_18_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_18_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_18_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_18_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_18_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_18_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_18_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_18_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_18_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_18_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_18_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_18_bottleneck_v2_add = resnet_v2_101_block3_unit_17_bottleneck_v2_add + resnet_v2_101_block3_unit_18_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_19_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_19_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_18_bottleneck_v2_add)
        resnet_v2_101_block3_unit_19_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_19_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_19_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_19_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_19_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_19_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_19_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_19_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_19_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_19_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_19_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_19_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_19_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_19_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_19_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_19_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_19_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_19_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_19_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_19_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_19_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_19_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_19_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_19_bottleneck_v2_add = resnet_v2_101_block3_unit_18_bottleneck_v2_add + resnet_v2_101_block3_unit_19_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_20_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_20_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_19_bottleneck_v2_add)
        resnet_v2_101_block3_unit_20_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_20_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_20_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_20_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_20_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_20_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_20_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_20_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_20_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_20_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_20_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_20_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_20_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_20_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_20_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_20_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_20_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_20_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_20_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_20_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_20_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_20_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_20_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_20_bottleneck_v2_add = resnet_v2_101_block3_unit_19_bottleneck_v2_add + resnet_v2_101_block3_unit_20_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_21_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_21_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_20_bottleneck_v2_add)
        resnet_v2_101_block3_unit_21_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_21_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_21_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_21_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_21_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_21_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_21_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_21_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_21_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_21_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_21_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_21_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_21_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_21_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_21_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_21_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_21_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_21_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_21_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_21_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_21_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_21_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_21_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_21_bottleneck_v2_add = resnet_v2_101_block3_unit_20_bottleneck_v2_add + resnet_v2_101_block3_unit_21_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_22_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_22_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_21_bottleneck_v2_add)
        resnet_v2_101_block3_unit_22_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_22_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_22_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_22_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_22_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_22_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_22_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_22_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_22_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_22_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_22_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block3_unit_22_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block3_unit_22_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_22_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_22_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block3_unit_22_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_22_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_22_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_22_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_22_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_22_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_22_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_22_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_22_bottleneck_v2_add = resnet_v2_101_block3_unit_21_bottleneck_v2_add + resnet_v2_101_block3_unit_22_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block3_unit_23_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block3_unit_23_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_22_bottleneck_v2_add)
        resnet_v2_101_block3_unit_23_bottleneck_v2_shortcut_MaxPool, resnet_v2_101_block3_unit_23_bottleneck_v2_shortcut_MaxPool_idx = F.max_pool2d(resnet_v2_101_block3_unit_22_bottleneck_v2_add, kernel_size=(1, 1), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        resnet_v2_101_block3_unit_23_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block3_unit_23_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block3_unit_23_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block3_unit_23_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block3_unit_23_bottleneck_v2_preact_Relu)
        resnet_v2_101_block3_unit_23_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_23_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_23_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block3_unit_23_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block3_unit_23_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_23_bottleneck_v2_Pad = F.pad(resnet_v2_101_block3_unit_23_bottleneck_v2_conv1_Relu, (1, 1, 1, 1), mode = 'constant', value = 0)
        resnet_v2_101_block3_unit_23_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block3_unit_23_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block3_unit_23_bottleneck_v2_Pad)
        resnet_v2_101_block3_unit_23_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block3_unit_23_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block3_unit_23_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block3_unit_23_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block3_unit_23_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block3_unit_23_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block3_unit_23_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block3_unit_23_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block3_unit_23_bottleneck_v2_add = resnet_v2_101_block3_unit_23_bottleneck_v2_shortcut_MaxPool + resnet_v2_101_block3_unit_23_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block4_unit_1_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block4_unit_1_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block3_unit_23_bottleneck_v2_add)
        resnet_v2_101_block4_unit_1_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block4_unit_1_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block4_unit_1_bottleneck_v2_shortcut_Conv2D = self.resnet_v2_101_block4_unit_1_bottleneck_v2_shortcut_Conv2D(resnet_v2_101_block4_unit_1_bottleneck_v2_preact_Relu)
        resnet_v2_101_block4_unit_1_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block4_unit_1_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block4_unit_1_bottleneck_v2_preact_Relu)
        resnet_v2_101_block4_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block4_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block4_unit_1_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block4_unit_1_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block4_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block4_unit_1_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block4_unit_1_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block4_unit_1_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block4_unit_1_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block4_unit_1_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block4_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block4_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block4_unit_1_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block4_unit_1_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block4_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block4_unit_1_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block4_unit_1_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block4_unit_1_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block4_unit_1_bottleneck_v2_add = resnet_v2_101_block4_unit_1_bottleneck_v2_shortcut_Conv2D + resnet_v2_101_block4_unit_1_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block4_unit_2_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block4_unit_2_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block4_unit_1_bottleneck_v2_add)
        resnet_v2_101_block4_unit_2_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block4_unit_2_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block4_unit_2_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block4_unit_2_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block4_unit_2_bottleneck_v2_preact_Relu)
        resnet_v2_101_block4_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block4_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block4_unit_2_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block4_unit_2_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block4_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block4_unit_2_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block4_unit_2_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block4_unit_2_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block4_unit_2_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block4_unit_2_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block4_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block4_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block4_unit_2_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block4_unit_2_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block4_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block4_unit_2_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block4_unit_2_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block4_unit_2_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block4_unit_2_bottleneck_v2_add = resnet_v2_101_block4_unit_1_bottleneck_v2_add + resnet_v2_101_block4_unit_2_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_block4_unit_3_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_101_block4_unit_3_bottleneck_v2_preact_FusedBatchNorm(resnet_v2_101_block4_unit_2_bottleneck_v2_add)
        resnet_v2_101_block4_unit_3_bottleneck_v2_preact_Relu = F.relu(resnet_v2_101_block4_unit_3_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_101_block4_unit_3_bottleneck_v2_conv1_Conv2D = self.resnet_v2_101_block4_unit_3_bottleneck_v2_conv1_Conv2D(resnet_v2_101_block4_unit_3_bottleneck_v2_preact_Relu)
        resnet_v2_101_block4_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block4_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(resnet_v2_101_block4_unit_3_bottleneck_v2_conv1_Conv2D)
        resnet_v2_101_block4_unit_3_bottleneck_v2_conv1_Relu = F.relu(resnet_v2_101_block4_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block4_unit_3_bottleneck_v2_conv2_Conv2D_pad = F.pad(resnet_v2_101_block4_unit_3_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_101_block4_unit_3_bottleneck_v2_conv2_Conv2D = self.resnet_v2_101_block4_unit_3_bottleneck_v2_conv2_Conv2D(resnet_v2_101_block4_unit_3_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_101_block4_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_101_block4_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(resnet_v2_101_block4_unit_3_bottleneck_v2_conv2_Conv2D)
        resnet_v2_101_block4_unit_3_bottleneck_v2_conv2_Relu = F.relu(resnet_v2_101_block4_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_101_block4_unit_3_bottleneck_v2_conv3_Conv2D = self.resnet_v2_101_block4_unit_3_bottleneck_v2_conv3_Conv2D(resnet_v2_101_block4_unit_3_bottleneck_v2_conv2_Relu)
        resnet_v2_101_block4_unit_3_bottleneck_v2_add = resnet_v2_101_block4_unit_2_bottleneck_v2_add + resnet_v2_101_block4_unit_3_bottleneck_v2_conv3_Conv2D
        resnet_v2_101_postnorm_FusedBatchNorm = self.resnet_v2_101_postnorm_FusedBatchNorm(resnet_v2_101_block4_unit_3_bottleneck_v2_add)
        resnet_v2_101_postnorm_Relu = F.relu(resnet_v2_101_postnorm_FusedBatchNorm)
        resnet_v2_101_pool5 = torch.mean(resnet_v2_101_postnorm_Relu, 3, True)
        resnet_v2_101_pool5 = torch.mean(resnet_v2_101_pool5, 2, True)
        resnet_v2_101_logits_Conv2D = self.resnet_v2_101_logits_Conv2D(resnet_v2_101_pool5)
        resnet_v2_101_SpatialSqueeze = torch.squeeze(resnet_v2_101_logits_Conv2D)
        MMdnn_Output_input = [resnet_v2_101_SpatialSqueeze]
        return MMdnn_Output_input


    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        if 'scale' in _weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(_weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(_weights_dict[name]['var']))
        return layer

# class ImageNet(Dataset):
#     """load data from img and csv"""
#     def __init__(self, dir, csv_path, transforms=None):
#         self.dir = dir
#         self.csv = pd.read_csv(csv_path)
#         self.transforms = transforms

#     def __getitem__(self, index):
#         img_obj = self.csv.loc[index]
#         ImageID = img_obj['ImageId'] + '.png'
#         Truelabel = img_obj['TrueLabel']
#         img_path = os.path.join(self.dir, ImageID)
#         pil_img = Image.open(img_path).convert('RGB')
#         if self.transforms:
#             data = self.transforms(pil_img)
#         else:
#             data = pil_img
#         return data, ImageID, Truelabel

#     def __len__(self):
#         return len(self.csv)

def load_ground_truth(csv_filename = '/home/lt/data/nips_1000/dev_dataset.csv'):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            if 'png' not in row['ImageId']:
                imgname = row['ImageId']+'.png'
            else:
                imgname = row['ImageId']
            image_id_list.append(imgname)
            if 'nips' in csv_filename:
                label_ori_list.append( int(row['TrueLabel']) - 1)
                label_tar_list.append( int(row['TargetClass']) - 1)
            else:
                label_ori_list.append( int(row['TrueLabel']) + 1)          # 原本torch模型类别0-999，现在加载的是tf模型，标签+1
                label_tar_list.append( int(row['TargetClass']) + 1)

    return image_id_list,label_ori_list,label_tar_list

class ImageNet(Dataset):
    # dirname 为训练/测试数据地址，使得训练/测试分开
    def __init__(self, adv_path, path = '/home/lt/vit_codes/PNA-PatchOut/dev_dataset.csv', transform = None):
        super(ImageNet, self).__init__()
        image_id_list,label_ori_list,label_tar_list = load_ground_truth(path)
        # '/home/lt/data/nips_1000/images/'
        self.paths = image_id_list
        self.true_labels = label_ori_list
        self.tar_labels = label_tar_list
        if transform == None:
            # self.transform = transforms_imagenet_wo_resize(params(model_name))
            self.transform = transforms.Compose([
                transforms.ToTensor(),]
            )
        else:
            self.transform = transform
        self.adv_path = adv_path
    def __getitem__(self, index):
        img_path, label, tar_label = os.path.join(self.adv_path, self.paths[index]), self.true_labels[index], self.tar_labels[index]
        img = Image.open(img_path).convert('RGB') 
        if self.transform is not None:
            img = self.transform(img) 
        return img, label, tar_label, self.paths[index]
    def __len__(self):
        return len(self.paths)

class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x

class TfNormalize(nn.Module):

    def __init__(self, mean=0, std=1, mode='tensorflow'):
        """
        mode:
            'tensorflow':convert data from [0,1] to [-1,1]
            'torch':(input - mean) / std
        """
        super(TfNormalize, self).__init__()
        self.mean = mean
        self.std = std
        self.mode = mode

    def forward(self, input):
        size = input.size()
        x = input.clone()

        if self.mode == 'tensorflow':
            x = x * 2.0 - 1.0  # convert data from [0,1] to [-1,1]
        elif self.mode == 'torch':
            for i in range(size[1]):
                x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x

    
def main():
    model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            Res101('/home/lt/code/ODI/models/tf_resnet_v2_101.npy').eval().cuda())

    transforms = T.Compose([
        T.Resize([299,299]),
        T.ToTensor()])  
    # img_path = '../../data/nips_1000/images/' 
    # csv_path = '../../data/nips_1000/dev_dataset.csv'
    
    img_path = '../../vit_codes/PNA-PatchOut/clean_resized_images' 
    csv_path = '../../vit_codes/PNA-PatchOut/dev_dataset.csv'
    
    data = ImageNet(img_path, csv_path, transforms)
    nips_loader = DataLoader(data, batch_size=20, shuffle=True, num_workers=4)
    
    acc = 0
    for imgs, labels, _,_ in tqdm(nips_loader):
        imgs = imgs.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            out = model(imgs)[0]
        pred = out.max(dim = 1)[1]
        acc += (pred == labels).sum().item()
    print(acc)

# 98.9%
if __name__ == "__main__":
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    main()

