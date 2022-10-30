import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randrange
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Activation
from tensorflow import keras
import os
from PIL import Image
from tensorflow.python.keras.backend import conv2d
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
label = []
shape1 = []
shape2 = []
def blur(image, Xlen,Ylen):
    #reduce a large image to a image of dimensions Xlen and Ylen
    output = image
    output = []

    for w in range(int(image.shape[1]/Xlen)):
        for l in range(int(image.shape[0]/Ylen)):
            count = 0
            for i in range(Xlen):
                for j in range(Ylen):
                    count = count + image[l*Ylen + j][w*Xlen + i]
            count = count/(Xlen*Ylen)
            try:
                output.append(int(count))
            except:
                output.append(image[l*Ylen + j][w*Xlen + i][0])
    output = np.asarray(output)
    output = output.reshape(int(image.shape[1]/Xlen),int(image.shape[0]/Ylen))
    return output
def separate(input):
    # specific to this program
    # turns string to list of ints
    output=[]
    for i in input:
        if i == '1':
            output.append(1)
        elif i=='0':
            output.append(0)
        elif i=='2':
            output.append(2)
    return output
def Counts(types, sequence):
    # counts occurances of [types] in [sequence]
    counts = [0 for i in types]
    for i in sequence:    
        for j in range(len(types)):
            if i == types[j]:
                counts[i] = counts[i]+1
    return types, counts
def eliminate(input, selections):
    # eliminates len(input)-selections from input
    output = input
    for i in range(len(input) - selections):
        random = randrange(0,len(output))
        output.pop(random)
    return output

def TakeShapes(input):
    # takes the shapes of the images specified by input and changes [shape1] and [shape2]
    for a,b,i in os.walk('C:/Users/anika/Downloads/ChestXRay/chest_xray/'):
        for n in i:
            if n in input:
                img = Image.open(os.path.abspath(os.path.join(a,n)))
                img = np.asarray(img)
                shape1.append(int(img.shape[0]/30))
                shape2.append(int(img.shape[1]/30))
def equalize(input, shape):
    # makes sure all images in input are same (shape)
    img = np.zeros(shape)
    print(input.shape)
    for h in range(input.shape[0]):
        for w in range(input.shape[1]):
            img[h][w] = input[h][w]
    return img 
    
def preprocessing():
    # index holds the file names label holds the binary values for pneumonia and normal
    index = []
    Label = []
    label 
    # this reads from the folder and populates and index
    train = "//chest_xray//"
    for a,b,i in os.walk(train):
        for j in i:
            index.append(j)
    bacteria = [index[i] for i in range(len(index)) if 'bacteria' in index[i]]
    virus = [index[i] for i in range(len(index)) if 'virus' in index[i]]
    Pneumonia = bacteria+virus
    Normal = [index[i] for i in range(len(index)) if not 'virus' in index[i] and not 'bacteria' in index[i]]
    plt.bar(['Pneumonia','Normal',],[len(Pneumonia),len(Normal)])
    plt.ylabel('no. of images')
    plt.xlabel('categories')
    plt.show()
    selectedN = eliminate(Normal, 1000)
    selectedP = eliminate(Pneumonia, 1000)

    order = selectedN+selectedP
    print(len(selectedN), len(selectedP))
    # stores and label to a file so we wont have to preprocess again later

    # this processes label and creates a graph


 

    # this part stores image array into a npy file 
    try:
        file2 = open("images.npy",'x')
    except:
        file2 = open("images.npy",'w')
    file2.close()
    X = []
    iter = 0 
    ter = 0
    TakeShapes(order)
    for a,b,i in os.walk('chest_xray/'):
        for n in i:
            if n in order:
                if 'virus' in n or 'bacteria' in n:
                    label.append(1)
                else:
                    label.append(0)
                iter = iter+1
                img = Image.open(os.path.abspath(os.path.join(a,n)))
                imgp = np.asarray(img)
                imgp = blur(imgp,30,30)
                plt.imshow(imgp)
                Cshape1 = max(shape1)
                Cshape2 = max(shape2)
                print(iter)
            
                img.close()
                item = equalize(imgp, (Cshape2,Cshape1))
                print(item.shape)
                X = np.append(X,item)
        X = np.asarray(X)
    np.save('images.npy',X)
    try:
        file = open("Labels.csv",'x')
    except:
        file = open("Labels.csv",'w')

    for i in range(len(label)):
        file.write(str(label[i]))
    file.close()
file = open("Labels.csv")
Label = file.read()
Label = separate(Label)
file.close()
shape1 = 97
shape2 = 87
XValues = np.load('images.npy')
print('LABEL: ',len(Label))
XValues = XValues.reshape(2000,shape1,shape2)
XTrain,XVal,YTrain,YVal = train_test_split(XValues,Label,test_size=0.25)
print('Train',len(XTrain),'Val',len(XVal))
XTrain = XTrain.reshape(1500,shape1,shape2,1)
XVal = XVal.reshape(500,shape1,shape2,1)
early_stop = EarlyStopping(monitor='val_loss',patience=2)
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3), activation='relu',input_shape=(shape1,shape2,1),padding='same'))#
model.add(BatchNormalization())#
model.add(MaxPool2D((2,2),strides=2,padding='same'))#
model.add(Dropout(0.30))#
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))#
model.add(MaxPool2D((2,2),padding='same'))#
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))#
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))#
model.add(Dropout(0.30))#
model.add(Flatten())#
model.add(Dense(150,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer ='adam',metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss',patience=2)
print(XTrain.shape, np.array(YTrain).shape)
model.fit(XTrain,np.array(YTrain),epochs=20,validation_data=(XVal,np.array(YVal)),callbacks=[early_stop])
losses=pd.DataFrame(model.history.history)


predictions = model.predict(XVal)

print(len([i for i in YTrain if i == 0]), len([i for i in YTrain if i == 1]))
truePositives = 0
for i in range(len(YVal)):
    if YVal[i] == 1 and predictions[i] > 0.5:
        truePositives = truePositives + 1
Positives = 0
for i in range(len(YVal)):
    if YTrain[i] == 1:
        Positives = Positives + 1
PredictedPositives = 0
for i in range(len(YVal)):
    if predictions[i] > 0.5:
        PredictedPositives = PredictedPositives + 1
print('Precision: ', truePositives/Positives)
print('recall: ', truePositives/PredictedPositives)

