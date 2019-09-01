import torch
import math
import os
import torch.nn as nn
import torch.nn.functional as fun
from torchsummary import summary
import torchvision
from collections import Counter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.misc import imread
from torch.optim import lr_scheduler
import pandas as pd
import time
import os
import csv
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


#Load Testing data

def readTestSigns(rootpath):
    images = [] # images
    labels = [] # corresponding labels
    gtFile = open(rootpath+ '/'+'GT-final_test.csv')
    gtReader = csv.reader(gtFile,delimiter=';')
    next(gtReader, None)
    count=0
    for row in gtReader:
        images.append(plt.imread(rootpath+'/'+row[0]))
        labels.append(row[7])
        print(count+1,end="\r")
        count+=1
    return  images,labels


def main:
    test_images,test_labels = readTestSigns(r'/home/konstantinoskk//Documents/linux_ip/Street/Test')
    print ('Test',len(test_labels), len(test_images))
