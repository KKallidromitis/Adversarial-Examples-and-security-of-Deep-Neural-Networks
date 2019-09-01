import torch
import torch.nn as nn
from collections import Counter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.misc import imread
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

#Load model
PATH = r'/home/konstantinoskk//Documents/linux_ip/model/model94.pth'
modelf=torchvision.models.resnet34(pretrained=True)
modelf.fc = nn.Linear(512, 43)
modelf.avgpool= nn.AdaptiveAvgPool2d(1)
modelf.load_state_dict(torch.load(PATH))
modelf=modelf.to(device)
print(modelf)

ts_images=torch.load(r'/home/konstantinoskk//Documents/linux_ip/Street/ts_images.pt)
ts_labels=torch.load(r'/home/konstantinoskk//Documents/linux_ip/Street/ts_labels.pt)
load_ts=torch.load(r'/home/konstantinoskk//Documents/linux_ip/Street/load_ts.pt)

resize=transforms.Compose([transforms.Resize((112,112)),transforms.ToTensor()])
transform = transforms.Compose([transforms.Resize((112,112)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
toPIL=torchvision.transforms.ToPILImage(mode=None)

#Evaluation of model and calculate class accuracy
correct = 0
total = 0
c=[]
class_correct=[[0]*43]
class_total=[[0]*43]
for i in range(len(ts_images)):
    inputs=ts_images[i]
    labels=ts_labels[i]
    inputs.unsqueeze_(0)
    labels.unsqueeze_(0)
    #print(i,inputs.shape,labels.shape)
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = modelf.eval()(inputs)
    out=outputs.cpu()
    out=out.permute(1,0)
    out=out.detach().numpy()
    out = [i[0] for i in out]
    softmax=np.exp(out)/sum(np.exp(out))
    c.append(max(softmax*100))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    class_total[0][labels]+=1
    if predicted==labels:
        class_correct[0][labels]+=1
    #print(labels,predicted)
    correct += (predicted == labels).sum().item()
print('Accuracy : %d %%' % (100 * correct / total))
c_acc=np.divide(class_correct,class_total)
print(c_acc)

#Select images for attacks
test_label=380
labs=[]
for i in range (len(ts_images)):
    labels=ts_labels[i]
    if int(labels)!=test_label:
        labs.append(i)
        t=toPIL(test_images[i])
        if c[i]<=100:
            if test_images[i].shape[0]>100:
                plt.imshow(t)
                print('ID',i,test_images[i].shape,c[i])
                if i==200:
                    break
                plt.show()
print(labs)
np.savetxt("/home/konstantinoskk/Documents/linux_ip/Street/adversarial_ex/labs.csv", labs)

#Generate attack types
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
    else:
        return 'Error'
    img=toPIL(img)
    plt.imshow(img)
    plt.show()
    correct=0
    img=transform(img)
    img.unsqueeze_(0)

    img,lab=img.to(device),lab.to(device)
    outputs = modelf.eval()(img)
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == lab).sum().item()
    print(predicted,correct)
    out=outputs.cpu()
    out=out.permute(1,0)
    out=out.detach().numpy()
    out = [i[0] for i in out]
    softmax=np.exp(out)/sum(np.exp(out))
    print('Attack '+str(num)+' Confidence',max(softmax*100))
    img=torch.reshape(img,(3,112,112))
    if enable==1:
        return img
    else:
        return

#Individual attacks
img=mpimg.imread(r'/home/konstantinoskk//Documents/linux_ip/Street/adversarial_ex/test_graffiti/00829.ppm')
lab=torch.tensor(2)
if 1 in attacks:
    g_attack(img,lab,1)
if 2 in attacks:
    g_attack(img,lab,2)
if 3 in attacks:
    g_attack(img,lab,3)

#Implementation of EOT
img=mpimg.imread(r'/home/konstantinoskk//Documents/linux_ip/Street/adversarial_ex/stick/o1.jpg')
lab=torch.tensor(14)

img=toPIL(img)
plt.imshow(img)
plt.show()
correct=0
img=transform(img)
img.unsqueeze_(0)
img,lab=img.to(device),lab.to(device)
outputs = modelf.eval()(img)
_, predicted = torch.max(outputs.data, 1)
correct = (predicted == lab).sum().item()
print(predicted,correct)
out=outputs.cpu()
out=out.permute(1,0)
out=out.detach().numpy()
out = [i[0] for i in out]
softmax=np.exp(out)/sum(np.exp(out))
print('Attack Confidence',max(softmax*100))
img=torch.reshape(img,(3,112,112))
