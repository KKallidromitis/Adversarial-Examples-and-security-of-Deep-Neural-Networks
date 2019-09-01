import torch
import math
import torch.nn as nn
import torch.nn.functional as fun
from torchsummary import summary
import torchvision
import pandas as pd
from collections import Counter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.misc import imread
from torch.optim import lr_scheduler
import time
import os
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

#Attack generation
def g_attack(img,lab,num):
    img=toPIL(img)
    img=resize(img)
    #print(img.shape)
    if num==1:
        img[:,75:85,25:85]=255
        img[:,25:35,45:55]=255
        img[:,50:70,25:35]=255
    elif num==2:
        img[:,50:60,30:70]=1
    elif num==3:
        img[:,75:80,65:85]=1
        img[:,25:35,40:50]=1
        img[:,50:65,25:35]=255
        img[:,55:70,65:75]=255
    if enable==1:
        return img
    else:
        return

#Apply to all images
import os
PATH=r'/home/konstantinoskk/Documents/linux_ip/Street/adversarial_ex/adv_tr/'

images=[]

for filename in os.listdir(PATH):
    filename=PATH+str(filename)
    images.append(plt.imread(filename))

for i in range (len(images)):
    images[i]=toPIL(images[i])
    images[i]=resize(images[i])
    #images[i]=images[i].unsqueeze_(0)
    print(i,end="\r")
print(images[0].shape)

images=torch.load(r'/home/konstantinoskk//Documents/linux_ip/Street/tr_images.pt')
resize=transforms.Compose([transforms.Resize((112,112)),transforms.ToTensor()])
transform = transforms.Compose([transforms.Resize((112,112)),transforms.ToTensor()])
toPIL=torchvision.transforms.ToPILImage(mode=None)
g_tr_images=torch.rand(len(images),3,112,112)
g_tr_images1=g_tr_images
g_tr_images2=g_tr_images
g_tr_images3=g_tr_images
attacks={1,2,3}
enable=1
count=0

#Generate attacks on seperate datasets
for img in images:
    lab=torch.tensor(14)
    if 1 in attacks:
        g_tr_images1[count]=g_attack1(img,lab)
    if 2 in attacks:
        g_tr_images2[count]=g_attack2(img,lab)
    if 3 in attacks:
        g_tr_images3[count]=g_attack3(img,lab)
    print(count,end=" ")
    count+=1
