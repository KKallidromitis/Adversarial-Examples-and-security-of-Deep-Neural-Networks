from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import pandas as pd
import time
import os
import csv
import copy
import torch
seed = 42
np.random.seed(seed)

#Training data
def readTrafficSigns(rootpath):
    images = [] # images
    labels = [] # corresponding labels
    sample = []
    count=0
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader, None)
        # loop over all images in current annotations file
        sam=0
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
            print(count+1,end="\r")
            count+=1
            if sam==200:
                #print(row)
                sample.append(plt.imread(prefix + row[0]))
            sam+=1
        gtFile.close()
    return images, labels, sample


#Testing data
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


train_images , train_labels , samples = readTrafficSigns(r'/home/konstantinoskk//Documents/linux_ip/Street/Train')
print ('Train',len(train_labels), len(train_images))

test_images,test_labels = readTestSigns(r'/home/konstantinoskk//Documents/linux_ip/Street/Test')
print ('Test',len(test_labels), len(test_images))

names=pd.read_csv(r'/home/konstantinoskk//Documents/linux_ip/Street/signnames.csv')
sign_names=np.array(names.iloc[:,1:3])

#Display sample images
def plot_imgs(samples):
    fig, axes = plt.subplots(nrows=3, ncols=15, figsize=(20,4))
    count1,count2=0,0
    toPIL=torchvision.transforms.ToPILImage(mode=None)
    resize = transforms.Resize((112,112))

    for s in samples:
        s=toPIL(s)
        s=resize(s)
        axes[count2][count1].imshow(s, interpolation='nearest', aspect='auto')
        axes[count2][count1].axis('off')
        count1+=1
        if count1==15:
            count1 = 0
            count2 +=1

            axes[2][13].axis('off')
            axes[2][14].axis('off')
            plt.show()

plot_imgs(samples)

#Measure bias in the dataset and imbalances between classes
label_counts = Counter(train_labels).most_common()
sign_names = [i[0] for i in sign_names]
for l, c in label_counts:
    print('Label count',c,'\t','Label:',l,str(sign_names[int(l)]))

#Create Validation Dataset
temp=Counter(train_labels).most_common()
order=[[] for _ in range(43)]
for l,c in temp:
    order[int(l)]=c

val_images=[]
val_labels=[]
tr_labels=[]
tr_images=[]
n1,n2=0,0
ratio=0.2

print(order)

for n in order:
    n2+=n
    #print(n1,n2)
    val_labels.extend(train_labels[n1:int(n1+n*ratio)])
    tr_labels.extend(train_labels[int(n1+n*ratio):n2])
    val_images.extend(train_images[n1:int(n1+n*ratio)])
    tr_images.extend(train_images[int(n1+n*ratio):n2])
    n1=n2
#print(val_labels)
ts_images=test_images
ts_labels=test_labels


#Shuffle function for images and labels
def shuffle(images,labels):
    s = np.random.choice(range(len(images)), len(images), replace=False)
    i=0
    temp_labels = labels
    temp_images = images
    for n in s:
        temp_labels[i]=labels[n]
        temp_images[i]=images[n]
        i+=1
    return temp_images,temp_labels

val_images,val_labels=shuffle(val_images,val_labels)
tr_images,tr_labels=shuffle(tr_images,tr_labels)

#Calculate average image ratio to avoid differences between images
av=0
for i in train_images:
    av+=i.shape[0]/i.shape[1]

for i in test_images:
    av+=i.shape[0]/i.shape[1]
av=av/(len(test_images)+len(train_images))
print('Average Ratio of Images',av)

#Declare transforms
transform = transforms.Compose([transforms.Resize((112,112)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
toPIL=torchvision.transforms.ToPILImage(mode=None)

#Dataset Preprocessing and save models
def Preprocessing(images,labels,int):
    print(images[0].shape)
    for i in range (len(labels)):
        images[i]=toPIL(images[i])
        images[i]=transform(images[i])
        images[i]=images[i].unsqueeze_(0)
        print(i,end="\r")
        print(images[0].shape)
        labels=list(map(int, labels))
        labels=torch.tensor(labels)
        print(labels[:10])
        images=torch.cat(images, dim=0, out=None)
        print(images.shape,labels.shape)
        batch_size=200
        workers=4
        td=torch.utils.data.TensorDataset(images,labels)
        load=torch.utils.data.DataLoader(td, batch_size=batch_size, num_workers=workers)
        print(load)
        torch.save(images, r'/home/konstantinoskk//Documents/linux_ip/Street/'+str(int)+'_images.pt')
        torch.save(labels, r'/home/konstantinoskk//Documents/linux_ip/Street/'+str(int)+'_labels.pt')
        torch.save(load, r'/home/konstantinoskk//Documents/linux_ip/Street/load_'+str(int)+'.pt')

Preprocessing(tr_images,tr_labels,"tr")
Preprocessing(val_images,val_labels,"val")
Preprocessing(ts_images,ts_labels,"ts")
