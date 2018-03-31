import numpy as np
import cv2
import matplotlib.pyplot as plt

#==============================================================================
test_f = open("test_images.txt", "r")
test_data = test_f.readlines()
test_f.close()

test_f2 = open("test_labels.txt", "r")
test_labels = test_f2.readlines()
test_f2.close()

train_f = open("train_images.txt", "r")
train_data = train_f.readlines()
train_f.close()

train_f2 = open("train_labels.txt", "r")
train_labels = train_f2.readlines()
train_f2.close()

test_data = test_data[0].split()
test_labels = test_labels[0].split()
train_data = train_data[0].split()
train_labels = train_labels[0].split()

temp_img = cv2.imread("img.png")
temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
temp_img = cv2.resize(temp_img, (28, 28))

print("Jinishpotro loaded successfully !!")
#======================================================================


data = test_data
labels = test_labels


def showImage(img):
    plt.imshow(img,cmap='gray',interpolation='bicubic')
    plt.show()
    
    return





arr = []
temp = []
cnt = 0

for i in range(0, len(data)):

    temp.append(int(data[i]))
    
    cnt+=1
    if cnt==784:
        arr.append(temp)
        temp = []
        cnt=0


        
f = open('Datasets/razikBhai_test.txt','w')

for i in range(0, len(arr)):
    
    arr2 = arr[i] 
    name = labels[i]
    
    
    print(name)
    
    for j in range(0, len(arr2)):
        now = arr2[j]/255
        if(now!=0):
            now=1.0
        f.write(str(now)+', ')
        

    f.write(str(name)+'\n')
    
f.close()