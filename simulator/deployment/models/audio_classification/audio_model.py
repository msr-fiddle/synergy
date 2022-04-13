import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math

######################################################################     
#Wwe will use a convolutional neural network to process
# the raw audio data. Usually more advanced transforms are applied to the
# audio data, however CNNs can be used to accurately process the raw data.
# The specific architecture is modeled after the M5 network architecture 
# described in https://arxiv.org/pdf/1610.00087.pdf. An important aspect
# of models processing raw audio data is the receptive field of their 
# first layer’s filters. Our model’s first filter is length 80 so when
# processing audio sampled at 8kHz the receptive field is around 10ms.
# This size is similar to speech processing applications that often use
# receptive fields ranging from 20ms to 40ms.
#
class AudioNet(nn.Module):
    def __init__(self, num_classes=10):  
        super(AudioNet, self).__init__()    
        self.conv1 = nn.Conv1d(1, 128, 80, 4)  
        self.bn1 = nn.BatchNorm1d(128)    
        self.pool1 = nn.MaxPool1d(4)    
        self.conv2 = nn.Conv1d(128, 128, 3)  
        self.bn2 = nn.BatchNorm1d(128)     
        self.pool2 = nn.MaxPool1d(4)       
        self.conv3 = nn.Conv1d(128, 256, 3)  
        self.bn3 = nn.BatchNorm1d(256)      
        self.pool3 = nn.MaxPool1d(4)     
        self.conv4 = nn.Conv1d(256, 512, 3) 
        self.bn4 = nn.BatchNorm1d(512)  
        self.pool4 = nn.MaxPool1d(4)    
        self.avgPool = nn.AvgPool1d(30) #input should be 512x30 so this outputs a 512x1   
        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, x): 
        x = self.conv1(x)    
        x = F.relu(self.bn1(x)) 
        x = self.pool1(x)   
        x = self.conv2(x)  
        x = F.relu(self.bn2(x))  
        x = self.pool2(x)   
        x = self.conv3(x)   
        x = F.relu(self.bn3(x))  
        x = self.pool3(x)  
        x = self.conv4(x)  
        x = F.relu(self.bn4(x))   
        x = self.pool4(x)  
        x = self.avgPool(x)   
        x = x.permute(0, 2, 1) #change the 512x1 to 1x512  
        x = self.fc1(x) 
        return F.log_softmax(x, dim = 2)       
   
