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
        g_tr_images1[count]=g_attack(img,lab)
    if 2 in attacks:
        g_tr_images2[count]=g_attack(img,lab)
    if 3 in attacks:
        g_tr_images3[count]=g_attack(img,lab)
    #print(count,end=" ")
    count+=1

#Load training Labels with pertubated images
tr_labels=torch.load(r'/home/konstantinoskk//Documents/linux_ip/Street/tr_labels.pt')
batch_size=200
workers=4
tr=torch.utils.data.TensorDataset(images,tr_labels)
load_tr1=torch.utils.data.DataLoader(tr, batch_size=batch_size, num_workers=workers)
print(load_tr1)

tr_images=images
epochs=10
total_step = len(tr_images)
print('Total step =',total_step,'Epochs =',epochs)

d=batch_size
step=(int((total_step/d)+1)*epochs)
print()
step_pe=int(step/epochs)
print('Number of steps per epoch' ,step_pe,'for batch =',d)

loss_list = np.zeros((step,1))
eloss=np.zeros((epochs,1))

#Train Neural Network mixing inside adversarial inputs
enable=1
def Net(model, criterion, optimizer, scheduler, num_epochs):
    start = time.time()
    tloss=0

    k,st=0,0
    vtemp=0

    for e in range(num_epochs):

        for i,(inputs, labels) in enumerate(load_tr1,0):

            if (i % 5==0):
                num=np.random.randint(3) #random number generation 0-2

                inputs=torch.reshape(inputs,(3,112,112))
                if num==0:
                    inputs=g_attack1(inputs,labels)
                elif num==1:
                    inputs=g_attack2(inputs,labels)
                elif num==2:
                    inputs=g_attack3(inputs,labels)
                inputs.unsqueeze_(0)
            #print(inputs.shape)
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            outputs = model(inputs)
            #print(labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            tloss = loss.item()
            ac=labels.shape[0]
            st+=ac
            inputs = inputs.to('cpu')
            labels = labels.to('cpu')

            print('Epoch:',e+1,'/',epochs,'Step:',st,'/',total_step,'Step Loss:',tloss, end="\r")
            loss_list[k,0]=tloss
            k+=1

        torch.save(modelf.state_dict(), r'/home/konstantinoskk//Documents/linux_ip/model/'+str(e)+'.pth')
        pe=step_pe*(e+1)
        eloss[e,0]= np.mean(loss_list[(pe-step_pe):(pe),0])
        print('----------------','Epoch:',e+1,'Loss:',round(eloss[e,0],10),'----------------')

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model

#Run training function
criterion = nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(modelf.parameters(), lr=0.01)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
modelf = Net(modelf, criterion, optimizer, exp_lr_scheduler,num_epochs=epochs)


#Load testing data to evaluate defence effectiveness
ts_images=torch.load(r'/home/konstantinoskk//Documents/linux_ip/Street/ts_images.pt)
ts_labels=torch.load(r'/home/konstantinoskk//Documents/linux_ip/Street/ts_labels.pt)
images=ts_images
g_ts_images1=torch.rand(len(images),3,112,112)
g_ts_images2=torch.rand(len(images),3,112,112)
g_ts_images3=torch.rand(len(images),3,112,112)

xat=[1,2,3]
enable=1

for x in xat:
    count=0
    for img in images:
        if x==1:
            g_ts_images1[count]=g_attack(img)
        if x==2:
            g_ts_images2[count]=g_attack(img)
        if x==3:
            g_ts_images3[count]=g_attack(img)
        print(count,end=" ")
        count+=1

ts_labels = torch.cat((ts_labels, ts_labels), 0)
g_ts_images1=torch.cat((g_ts_images1, ts_images), 0)
g_ts_images2=torch.cat((g_ts_images2, ts_images), 0)
g_ts_images3=torch.cat((g_ts_images3, ts_images), 0)

#Test accuracy on trained model
batch_size=1
workers=4
l=[1,2,3]
UNIV= r'/home/konstantinoskk//Documents/linux_ip/model/'
model=torchvision.models.resnet34(pretrained=True)
model.fc = nn.Linear(512, 43)
model.avgpool= nn.AdaptiveAvgPool2d(1)
model.load_state_dict(torch.load(UNIV+'model94.pth'))
model=model.to(device)



def testacc(model,load_ts):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs,labels in load_ts:
            if (i % 5==0):
                num=np.random.randint(3)
                inputs=torch.reshape(inputs,(3,112,112))
                if num==0:
                    inputs=g_attack1(inputs,labels)
                elif num==1:
                    inputs=g_attack2(inputs,labels)
                elif num==2:
                    inputs=g_attack3(inputs,labels)
                inputs.unsqueeze_(0)

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            #print(inputs.shape)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


        print(correct,total)
    return (100 * correct / total)

#Test model for different training samples
def load_model(PATH,ts_images,ts_labels):
    accuracies=np.zeros((1,10))

    for filename in os.listdir(PATH):
        c=int(str(filename)[0])
        filename=str(PATH+str(filename))
        model=torchvision.models.resnet34(pretrained=True)
        model.fc = nn.Linear(512, 43)
        model.avgpool= nn.AdaptiveAvgPool2d(1)
        print(filename)
        model.load_state_dict(torch.load(filename))
        model=model.to(device)

        ts=torch.utils.data.TensorDataset(ts_images,ts_labels)
        load_ts=torch.utils.data.DataLoader(ts, batch_size=batch_size, num_workers=workers)

        accuracies[0,c]=(testacc(model,load_ts))
    return accuracies

for x in l:
    if x==1:
        PATH=UNIV+'a1/'
        a=load_model(PATH,g_ts_images1,ts_labels)
    if x==2:
        PATH=UNIV+'a2/'
        b=load_model(PATH,g_ts_images2,ts_labels)
    if x==3:
        PATH=UNIV+'a3/'
        c=load_model(PATH,g_ts_images3,ts_labels)

#PLot accuracy/loss results
epochs=10
e=np.reshape((np.linspace(1,epochs, num=epochs)),(-1,1))
a=np.reshape(a,(10,1))
b=np.reshape(b,(10,1))
c=np.reshape(c,(10,1))
plt.plot(e,a,label='attack 1',color='black')
plt.plot(e,b,label='attack 2',dashes=[3,3,2,2])
plt.plot(e,c,label='attack 3',dashes=[3,8],color='grey')
plt.legend(loc='top-right')
plt.xlabel('Epochs')
plt.ylabel('Accuracy of Adversarial Examples')
plt.show()
