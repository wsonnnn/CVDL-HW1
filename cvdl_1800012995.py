#导入依赖的库
import keras
import numpy as np
import math
import os
import string
import csv
import cv2
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.initializers import he_normal
from keras.layers import Dense,Input,add,Activation,Lambda,concatenate
from keras.layers import Conv2D,AveragePooling2D,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras import optimizers,regularizers
from keras.callbacks import LearningRateScheduler,TensorBoard
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model


#声明所需要的常量
train_path=r"C:/Users/sll82/Downloads/bird/bird/train/train/"
train_type_path=r"C:/Users/sll82/Downloads/bird/bird/train.csv"
valid_path=r"C:/Users/sll82/Downloads/bird/bird/valid/valid/"
valid_type_path=r"C:/Users/sll82/Downloads/bird/bird/valid.csv"
test_num=24497
valid_num=900
img_rows=224
img_cols=224
img_channels=3
classes=180
batch_size=16
weight_decay=1e-4
compression=0.5
growth_rate=12
epochs=185
iterations=400
log_path="./densenet_save"

#声明函数

#加载图片
def load_data(path,num):
    data=[]
    for i in range(num):
        string=str(i).zfill(5)
        img=cv2.imread(path+string+'.jpg')
        data.append(img)
    return data
#加载类型数据
def load_type(path):
    y=[]
    with open(path,"r",encoding="utf-8") as f:
        reader=csv.reader(f)
        #print(type(reader))
        i=0
        for row in reader:
            if i==0:
                i+=1
                continue
            i+=1
            #print(row[1])
            y.append(int(row[1]))
    return y
#图片预处理
def color_prefit(x):
    x=x.astype('float32')
    for i in range(3):
        x[:,:,:,i]=(x[:,:,:,i]-np.mean(x[:,:,:,i]))/np.std(x[:,:,:,i])
    return x
#调整学习率
def scheduler(epoch):
    if epoch<60:
        return 0.1
    if epoch<110:
        return 0.01
    if epoch<150:
        return 0.001
    else:
        return 0.0001
#卷积层，使用正则化
def conv(x,out_filters,k_size):
    return Conv2D(filters=out_filters,
                  kernel_size=k_size,
                  strides=(1,1),
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(weight_decay),
                  use_bias=False)(x)

def BN_ReLU(x):
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    return x

def bottleneck(x):
    channels = growth_rate * 4
    x = bn_relu(x)
    x = conv(x, channels, (1,1))
    x = bn_relu(x)
    x = conv(x, growth_rate, (3,3))
    return x

def mid_trans(x, inputchannels):
    outputchannels = int(inputchannels * compression)
    x = BN_ReLU(x)
    x = conv(x, outputchannels, (1,1))
    x = AveragePooling2D((2,2), strides=(2, 2))(x)
    return x, outputchannels

def dense_block(x,blocks,nchannels):
    concat = x
    for i in range(blocks):
        x = bottleneck(concat)
        concat = concatenate([x,concat], axis=-1)
        nchannels += growth_rate
    return concat, nchannels

def densenet(img_input,classes_num):
    nchannels=growth_rate*2
    x=conv(img_input,nchannels,(3,3))
    x,nchannels=dense_block(x,8,nchannels)
    x,nchannels=mid_trans(x,nchannels)
    x,nchannels=dense_block(x,8,nchannels)
    x,nchannels=mid_trans(x,nchannels)
    x,nchannels=dense_block(x,8,nchannels)
    x=BN_ReLU(x)
    x=GlobalAveragePooling2D()(x)
    x=Dense(units=10,
                 activation='softmax',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(x)
    return x


#加载存储数据
train_x=np.array(load_data(train_path))
np.save('bird_data.npy',train_x)

train_y=np.array(load_type(train_type_path))
np.save('bird_data_type.npy',train_y)

valid_x=np.array(load_data(valid_path,valid_num))
np.save('bird_valid.npy',valid_x)

valid_y=np.array(load_type(valid_type_path))
np.save('bird_valid_type.npy',valid_y)


#载入数据
train_x=np.load('bird_data.npy')
train_y=np.load('bird_data_type.npy')
valid_x=np.load('bird_valid.npy')
valid_y=np.load('bird_valid_type.npy')
#类型->onehot向量
y_train=keras.utils.to_categorical(train_y,classes)
y_valid=keras.utils.to_categorical(valid_y,classes)
#图片预处理
x_train=color_prefit(train_x)
x_valid=color_prefit(valid_x)
#声明模型
img_input=Input(shape=(img_rows,img_cols,img_channels))
output=densenet(img_input,classes)
model=Model(img_input,output)
#print(model.summary())

#优化器使用sgd
sgd=optimizers.SGD(lr=.1,momentum=0.9,nesterov=True)

#损失函数使用categorical_crossentropy，此外还试用了KL散度
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

#使用tensorboard进行数据可视化
tb_cb=TensorBoard(log_dir=log_path,histogram_freq=0)
change_lr=LearningRateScheduler(scheduler)
cbks=[change_lr,tb_cb]

#对预处理后的图像进行数据增强
datapowerful=ImageDataGenerator(horizontal_flip=True,
                          width_shift_range=0.125,
                           vertical_flip=True,
                          height_shift_range=0.125,
                          fill_mode='constant',
                          cval=0.)
datagenpowerful.fit(x_train)

#两种训练模型的方式，1 直接将全部图片加载到内存中，2 随用随取
model.fit_generator(datapowerful.flow(x_train,_train,batch_size=batch_size),
                   steps_per_epoch=iterations,
                   epochs=epochs,
                   callbacks=cbks,
                   validation_data=(x_valid,y_valid))

#这种方式需要将数据增强的方式在flow_from_directory中声明
model.fit_generator(datapowerful.flow_from_directory(direction="./bird",
                    target_size=(224,224),class_mode='categorical',
                    batch_size=batch_size,featurewise_std_normalization=True,
                    rotation_range=0.1,width_shift_range=0.1,
                    height_shift_range=0.1),
                   steps_per_epoch=iterations,
                   epochs=epochs,
                   callbacks=cbks,
                   validation_data=(x_valid,y_valid))

model.save('densenet.h5')


#读取训练好的模型，并对test数据进行预测
path="C:/Users/sll82/Downloads/densenet(2).h5"

model=load_model(path)
path2="C:/Users/sll82/Downloads/bird/bird/test/test/"
head=["ID","Category"]
rows=[]
for i in range(900):
    img=cv2.imread(path2+str(i).zfill(5)+".jpg")
    img=img.reshape(1,224,224,3).astype('float32')
    for u in range(3):
        img[:,:,:,u]=(img[:,:,:,u]-np.mean(img[:,:,:,u]))/np.std(img[:,:,:,u])
    predict = model.predict(img)
    predict=np.argmax(predict,axis=1)[0]
    res={}
    res['ID']=str(i).zfill(5)
    res['Category']=predict
    print(res)
    rows.append(res)

with open('res'+'sec'+'.csv','w',newline='',encoding='utf-8')as f2:
    f_csv = csv.DictWriter(f2,head)
    f_csv.writeheader()
    f_csv.writerows(rows)