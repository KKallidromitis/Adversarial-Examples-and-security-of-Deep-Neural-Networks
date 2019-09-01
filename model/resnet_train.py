import torch
import os
import torch.nn as nn
import torch.nn.functional as fun
from torchsummary import summary
import torchvision
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import pandas as pd
import time
import os
import csv
import copy

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Load datasets
tr_images=torch.load(r'/home/konstantinoskk//Documents/linux_ip/Street/tr_images.pt)
val_images=torch.load(r'/home/konstantinoskk//Documents/linux_ip/Street/val_images.pt)
ts_images=torch.load(r'/home/konstantinoskk//Documents/linux_ip/Street/ts_images.pt)

load_tr=torch.load(r'/home/konstantinoskk//Documents/linux_ip/Street/load_tr.pt)
load_val=torch.load(r'/home/konstantinoskk//Documents/linux_ip/Street/load_val.pt)
load_ts=torch.load(r'/home/konstantinoskk//Documents/linux_ip/Street/load_ts.pt)

#Check conv layers with torchvision
model=torchvision.models.resnet34()
model.avgpool= nn.AdaptiveAvgPool2d((1))
model.fc = nn.Linear(512, 43)
model= model.to(device)
summary(model, (3, 112, 112))

#Initialise network
modelf=torchvision.models.resnet34(pretrained=True)
for param in modelf.parameters():
    param.requires_grad = True
modelf.fc = nn.Linear(512, 43)
modelf.avgpool= nn.AdaptiveAvgPool2d(1)
print(modelf)
modelf=modelf.to(device)

#initialise parameters
epochs=30
total_step = len(tr_images)
print('Total step =',total_step,'Epochs =',epochs)

d=batch_size
step=(int((total_step/d)+1)*epochs)
print()
step_pe=int(step/epochs)
print('Number of steps per epoch' ,step_pe,'for batch =',d)

loss_list = np.zeros((step,1))
eloss=np.zeros((epochs,1))
tacc=[]
vacc=[]
vloss=np.zeros((epochs,1))
val_step = len(val_images)
test_step = len(ts_images)

#Train model
def Net(model, criterion, optimizer, scheduler, num_epochs):
    start = time.time()
    tloss=0

    k,st=0,0
    vtemp=0

    for e in range(num_epochs):
        correct = 0
        total = 0
        for i,(inputs, labels) in enumerate(load_tr,0):

            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
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
            tloss=0

            if st==total_step:
                tac=(100 * correct / total)
                tacc.append(tac)
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs,labels in load_val:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = modelf(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        vtloss = criterion(outputs, labels)
                        vtloss = vtloss.item()
                        vloss[e,0]+=vtloss
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    vac=(100 * correct / total)
                    vloss[e,0]=vloss[e,0]/(len(val_images)/d)
                    vacc.append(vac)
                if vac>=vtemp:
                    torch.save(modelf.state_dict(), r'/home/konstantinoskk//Documents/linux_ip/model/model.pth')
                    vtemp=vac
                    ef=e+1
        st=0
        pe=step_pe*(e+1)
        eloss[e,0]= np.mean(loss_list[(pe-step_pe):(pe),0])
        print('----------------','Epoch:',e+1,'Loss:',round(eloss[e,0],10),'Tac',tac,'Vac',vac,'----------------')

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Final Model produced at Epoch',ef)
    return model

#Run model
criterion = nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(modelf.parameters(), lr=0.01)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
modelf = Net(modelf, criterion, optimizer, exp_lr_scheduler,num_epochs=epochs)
torch.save(modelf.state_dict(), r'/home/konstantinoskk//Documents/linux_ip/model/model_last.pth')

#Plot loss and accuracy
e=np.reshape((np.linspace(1,step, num=step)),(-1,1))
plt.plot(e,loss_list,label='step trainning loss')
plt.show()

e=np.reshape((np.linspace(1,epochs, num=epochs)),(-1,1))
plt.plot(e,eloss,label='trainning loss',color='black')
plt.plot(e,vloss,label='validation loss',dashes=[3,3,2,2])
plt.legend(loc='top-right')
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy Loss')
plt.show()

epochs=5
a=[12.63,88.28,88.71,89.23,93.27]
b=[12.63,94.22,92.43,93.66,92.61]
e=np.reshape((np.linspace(1,epochs, num=epochs)),(-1,1))
plt.plot(e,a,label='Training /5',color='black')
plt.plot(e,b,label='Traning /1',dashes=[3,3,2,2])
plt.legend(loc='top-right')
plt.xlabel('Epochs')
plt.ylabel('Testing Accuracy')
plt.show()

#Testing Accuracy
correct = 0
total = 0
with torch.no_grad():
    for inputs,labels in load_ts:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = modelf(inputs)
        #print(inputs.shape)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        #print(labels,predicted,(predicted == labels).sum().item())
print('Accuracy : %d %%' % (100 * correct / total))
