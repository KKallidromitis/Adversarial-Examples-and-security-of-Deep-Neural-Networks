#Additional Dependencies:
#https://github.com/ChiWeiHsiao/SphereNet-pytorch
#https://github.com/bethgelab/foolbox

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import argparse
import numpy as np
import zipfile

from dataset import ImageDataset
from net_sphere import sphere20a #Uses the spherenet neural architecture
import imp
import torchvision
import foolbox

print(np.version.version)
net=sphere20a()
net.load_state_dict(torch.load('model/sphere20a_20171020.pth'))
#net.cuda()
net.eval()
net.feature = True

model=foolbox.models.PyTorchModel(net, (0,1), 512, preprocessing=(0, 1))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
IMAGES="/home/konstantinoskk/Documents/Bench/test/"
img=mpimg.imread(IMAGES+'Alexa_Loren_0001.jpg')

import torchvision.transforms as transforms
resize1=transforms.Resize((112,96))
randcrop=transforms.RandomCrop((112,96))
grey=transforms.Grayscale(num_output_channels=1)
transform = transforms.ToTensor()
toPIL=torchvision.transforms.ToPILImage(mode=None)

im=toPIL(img)
im=resize1(im)
plt.imshow(im)
im=transform(im)
print(im.shape)

im=np.asarray(im)
print(im.shape)
np.argmax(model.predictions(im))


#attack = foolbox.attacks.FGSM(model)
criterion=foolbox.criteria.Misclassification()
attack=foolbox.attacks.SaltAndPepperNoiseAttack(model=model, criterion=criterion,threshold=0.01)
adversarial = attack(im,218)

a=transform(adversarial)
a=a.permute(1,0,2)
a=a.transpose(1,2)
print(a.shape)
a=toPIL(a)
plt.imshow(a)

a1=np.asarray(a)
blur_image = cv2.medianBlur(a1,15)
gg=np.reshape(blur_image,(3,112,96))
bi=torch.as_tensor(blur_image)
bi=bi.permute(2,0,1)
print(bi.shape)
bi=toPIL(bi)

plt.imshow(bi)
np.argmax(model.predictions(adversarial))
diff=gg-im
print(diff.shape)
d=transform(diff)
d=d.permute(1,0,2)
d=d.transpose(1,2)
d=toPIL(d)
plt.imshow(d)

#Attacks on Multiple Images
def load_img(PATH):
    images=[]
    names=[]
    for filename in os.listdir(PATH):
        filename=PATH+str(filename)
        print(filename)
        img=mpimg.imread(filename)
        images.append(img)
        names.append(filename)
    return images,names

def import_img(PATH):
    images=[]
    labels=[]
    names=[]
    count=0
    for filename in os.listdir(PATH):
        filename=PATH+str(filename)+'/'
        img,name=load_img(filename)
        images.extend(img)
        names.extend(name)
        lab= [count]*len(img)
        labels.extend(lab)
        print('{0}\r'.format(count))
        count+=1
    return images,labels,names

LOAD="/home/konstantinoskk/Documents/face/lfw/"
images,labels,names=import_img(LOAD)

for i in range (len(images)):
    images[i]=toPIL(images[i])
    images[i]=transform(images[i])
    images[i]=images[i].unsqueeze_(0)
images=torch.cat(images, dim=0, out=None)

for i in range (len(names)):
    names[i]=names[i].replace('lfw','lft')


images_p=[]
for i in range(len(images)):
    im=images[i]
    im=toPIL(im)
    #plt.imshow(im)
    #plt.show()
    im=resize1(im)
    im=transform(im)
    im=np.asarray(im)
    #print(im.shape)
    lab=np.argmax(model.predictions(im))
    adversarial = attack(im,lab)
    #print(adversarial.shape)
    images_p.append(adversarial)
    ad=transform(adversarial)
    #print(a.shape)
    ad=ad.permute(1,0,2)
    ad=ad.transpose(1,2)
    ad=toPIL(ad)
    print(i,names[i])
    ad.save(names[i], "JPEG", quality=100, optimize=True, progressive=True)
