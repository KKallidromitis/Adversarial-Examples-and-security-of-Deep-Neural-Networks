#Script to convert images from ppm to jpg
import pandas as pd
import numpy as np
import csv
import os
labels=pd.read_csv(r'/home/konstantinoskk/Documents/linux_ip/Street/adversarial_ex/labs.csv',header=None)
index=[]
for i, row  in labels.iterrows():
    index.append(int(row))
print(index)


rc={} #Hashmap datastructure for faster search
for i in range (len(index)):
    if int(index[i])<= 10000:
        index[i]='0'+str(index[i])
        if int(index[i])<= 1000:
            index[i]='0'+str(index[i])
        if int(index[i])<= 100:
            index[i]='0'+str(index[i])
    else:
        index[i]=str(index[i])
    index[i]=str(index[i])+'.ppm'
    rc[index[i]]=1
print(index)

#Run conversion
PATH=r'/home/konstantinoskk/Documents/linux_ip/Street/Test'
DEST=r'/home/konstantinoskk/Documents/linux_ip/Street/adversarial_ex/adv_tr/test_stop_signs'
count=0
for filename in os.listdir(PATH):
    if filename in rc:
        file=str(filename)
        file=file[:5]
        file=str(DEST)+str(file)
        filename=PATH+str(filename)
        print(filename)
        #print(file)
        !convert echo {filename} to echo {file}.jpg

#Training Images
PATH=r'/home/konstantinoskk/Documents/linux_ip/Street/Train/00014/'
DEST=r'/home/konstantinoskk/Documents/linux_ip/Street/adversarial_ex/adv_tr/train_stop_signs/'
for filename in os.listdir(PATH):
    file=str(filename)
    file=file[:11]
    file=str(DEST)+str(file)
    filename=PATH+str(filename)
    print(filename)
    print(file)
    !convert echo {filename} to echo {file}.jpg
