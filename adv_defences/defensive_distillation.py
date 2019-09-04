#Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

import torch
import os
import torch.nn as nn
import torch.nn.functional as fun
from torch.autograd import Variable
from torch import autograd
from torchsummary import summary
import torchvision
from collections import Counter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
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

#Load datasets
tr_images=torch.load(r'/home/konstantinoskk//Documents/linux_ip/Street/tr_images.pt)
val_images=torch.load(r'/home/konstantinoskk//Documents/linux_ip/Street/val_images.pt)
ts_images=torch.load(r'/home/konstantinoskk//Documents/linux_ip/Street/ts_images.pt)

load_tr=torch.load(r'/home/konstantinoskk//Documents/linux_ip/Street/load_tr.pt)
load_val=torch.load(r'/home/konstantinoskk//Documents/linux_ip/Street/load_val.pt)
load_ts=torch.load(r'/home/konstantinoskk//Documents/linux_ip/Street/load_ts.pt)

#Custom Loss function
def softmax(x,T):
    if x.shape[0]>1:
        soft=torch.zeros([x.shape[0], x.shape[1]])
        for i in range (x.shape[0]):
            e_x = torch.exp((x[i] - torch.max(x[i]))/T)
            temp=(e_x / torch.sum(e_x))
            temp=torch.as_tensor(temp)
            soft[i,:]=temp
    else:
        e_x = torch.exp((x - torch.max(x))/T)
        soft=(e_x / torch.sum(e_x))
    return torch.log(soft)

def onehotencoding(vec,dim):
    res=np.zeros((len(vec),dim))
    for row in range (len(vec)):
        res[row,vec[row]]=1
    res=torch.as_tensor(res)
    return res

class CrossEntropyLoss(torch.nn.Module):

    def __init__(self):
        super(CrossEntropyLoss1,self).__init__()

    def forward(self,pred,soft_targets):
        one=onehotencoding(soft_targets,43).float()
        one.requires_grad_(True)
        mult=Variable(pred, requires_grad=True)
        mult=(softmax(mult,1)) #40
        mult= one* mult
        mult=torch.sum(mult, 1)
        mult=torch.log(mult)
        loss=-torch.mean(mult)
        return loss

#Training model
def Net(model, criterion, optimizer, scheduler, num_epochs):
    start = time.time()
    tloss=0
    k,st=0,0
    vtemp=0
    for e in range(num_epochs):

        for i,(inputs, labels) in enumerate(load_tr,0):

            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            outputs=fun.log_softmax(outputs/40, dim=-1)
            ll=outputs.cpu().detach().numpy()
            ll=torch.as_tensor(ll)
            #ll=ll.to(device).long()
            #print(outputs.shape,ll.shape)

            outputs=outputs.to(device)
            loss=criterion(outputs,labels)

            loss.backward()
            optimizer.step()

            tloss = loss.item()
            ac=labels.shape[0]
            st+=ac
            inputs = inputs.to('cpu')
            labels = labels.to('cpu')

            print('Epoch:',e+1,'/',epochs,'Step:',st,'/',total_step,'Step Loss:',tloss,end="\r")
            loss_list[k,0]=tloss
            k+=1
            tloss=0

            if st==total_step:
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs,labels in load_val:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = modelf(inputs)


                        _, predicted = torch.max(outputs.data, 1)

                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    vac=(100 * correct / total)
                if vac>vtemp:
                    torch.save(modelf.state_dict(), r'/home/konstantinoskk//Documents/linux_ip/model/model_40.pth')
                    vtemp=vac
                    ef=e+1
        st=0
        pe=step_pe*(e+1)
        eloss[e,0]= np.mean(loss_list[(pe-step_pe):(pe),0])
        print('----------------','Epoch:',e+1,'Loss:',round(eloss[e,0],10),'Vac',round(vac,2),'----------------')

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Final Model produced at Epoch',ef)
    return model

criterion = nn.NLLLoss()
optimizer =torch.optim.Adam(modelf.parameters(), lr=0.01)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
modelf = Net(modelf, criterion, optimizer, exp_lr_scheduler,num_epochs=epochs)
torch.save(modelf.state_dict(), r'/home/konstantinoskk//Documents/linux_ip/model/model_last_40.pth')


#Generate New labels
PATH = r'/home/konstantinoskk//Documents/linux_ip/model/model_40.pth'
modelf=torchvision.models.resnet34(pretrained=True)
modelf.fc = nn.Linear(512, 43)
modelf.avgpool= nn.AdaptiveAvgPool2d(1)
modelf.load_state_dict(torch.load(PATH))
modelf=modelf.to(device)
batch_size=200
workers=4
tr=torch.utils.data.TensorDataset(tr_images,tr_labels)
load_tr=torch.utils.data.DataLoader(tr, batch_size=batch_size, num_workers=workers)
print(load_tr)

correct = 0
total = 0
count=[0,200]
new_labels_tr=np.zeros((len(tr_labels),43))

with torch.no_grad():
    for inputs,labels in load_tr:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = modelf(inputs)
        print(outputs.shape)
        ll=outputs.cpu().detach().numpy()
        new_labels_tr[count[0]:count[1],:]=ll
        count[0]+=200
        count[1]+=200
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        #print(labels,predicted,(predicted == labels).sum().item())
print('Accuracy : %d %%' % (100 * correct / total))
new_labels_tr=torch.as_tensor(new_labels_tr)
new_labels_tr=fun.softmax(new_labels_tr/40,1)
#store new_labels
torch.save(new_labels_tr,r'/home/konstantinoskk//Documents/linux_ip/Street/new_labels_tr.pt')

#Neural network 2
criterion = nn.BCELoss()
optimizer =torch.optim.Adam(modelf.parameters(), lr=0.01)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
modelf = Net(modelf, criterion, optimizer, exp_lr_scheduler,num_epochs=epochs)
torch.save(modelf.state_dict(), r'/home/konstantinoskk//Documents/linux_ip/model/model_last_40_p2.pth')

#Testing accuracy
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


#Test effectiveness
PATH = r'/home/konstantinoskk//Documents/linux_ip/model/model_40.pth'
model1=torchvision.models.resnet34(pretrained=True)
model1.fc = nn.Linear(512, 43)
model1.avgpool= nn.AdaptiveAvgPool2d(1)
model1.load_state_dict(torch.load(PATH))
model1=model1.to(device)

PATH = r'/home/konstantinoskk//Documents/linux_ip/model/model_40_p2.pth'
model2=torchvision.models.resnet34(pretrained=True)
model2.fc = nn.Linear(512, 43)
model2.avgpool= nn.AdaptiveAvgPool2d(1)
model2.load_state_dict(torch.load(PATH))
model2=model2.to(device)

resize=transforms.Compose([transforms.Resize((112,112)),transforms.ToTensor()])
transform = transforms.Compose([transforms.Resize((112,112)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
toPIL=torchvision.transforms.ToPILImage(mode=None)

def labelsgen(model,s,inputs,labels):
    with torch.no_grad():
        inputs=toPIL(inputs)
        inputs=resize(inputs)
        inputs.unsqueeze_(0)
        inputs, labels = inputs.to(device), labels.to(device)
        if model==1:
            PATH = r'/home/konstantinoskk//Documents/linux_ip/model/model_5.pth'
            model=torchvision.models.resnet34(pretrained=True)
            model.fc = nn.Linear(512, 43)
            model.avgpool= nn.AdaptiveAvgPool2d(1)
            model.load_state_dict(torch.load(PATH))
            model=model.to(device)
        else:
            PATH = r'/home/konstantinoskk//Documents/linux_ip/model/model_40_p2.pth'
            model=torchvision.models.resnet34(pretrained=True)
            model.fc = nn.Linear(512, 43)
            model.avgpool= nn.AdaptiveAvgPool2d(1)
            model.load_state_dict(torch.load(PATH))
            model=model.to(device)


        outputs = model(inputs)
        ll=outputs.cpu().detach().numpy()
        new_labels=outputs
        new_labels=fun.softmax(new_labels/s,1)

        return new_labels

initial=labelsgen(1,5,img,torch.as_tensor([12]))
final=labelsgen(model2,1,img,initial)

n_digits = 18
a= torch.round(initial * 10**n_digits) / (10**n_digits)
print(a)

b= torch.round(final * 10**n_digits) / (10**n_digits)
print(b)
