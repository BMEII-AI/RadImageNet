#!/usr/bin/env python
# coding: utf-8


import argparse
parser = argparse.ArgumentParser()



parser = argparse.ArgumentParser()
parser.add_argument('--gpu_node', type=str, help='specify gpu nodes')
parser.add_argument('--database', type=str, help='choose RadImageNet or ImageNet')
parser.add_argument('--model_name', type=str, help='choose IRV2/ResNet50/DenseNet121/InceptionV3')
parser.add_argument('--batch_size', type=int, help='batch size', default=256)
parser.add_argument('--image_size', type=int, help='image size', default=256)
parser.add_argument('--epoch', type=int, help='number of epochs', default=30)
parser.add_argument('--structure', type=str, help='unfreezeall/freezeall/unfreezetop10', default=30)
parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
args = parser.parse_args()



import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2, ResNet50, DenseNet121, VGG19, InceptionV3
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,Callback, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import regularizers, activations
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import AUC
import os
from time import time
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime




### Limit to the first GPU for this model  
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_node


### Import pre-trained weights from ImageNet or RadImageNet
database = args.database
if not database in ['RadImageNet', 'ImageNet']:
    raise Exception('Pre-trained database not exists. Please choose ImageNet or RadImageNet')
    
if not args.structure in ['unfreezeall', 'freezeall','unfreezetop10']:
    raise Exception('Freeze any layers? Choose to unfreezeall/freezeall/unfreezetop10 layers for the network.')

### Set up training image size, batch size and number of epochs
image_size = args.image_size
batch_size = args.batch_size
num_epoches = args.epoch



# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


train_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                    rescale=1./255,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=0.1,
                                    fill_mode='nearest')



data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)




df_train = pd.read_csv("dataframe/train.csv")
df_val = pd.read_csv("dataframe/val.csv")





train_generator = train_data_generator.flow_from_dataframe(
        dataframe=df_train,
        x_col = 'filename',
        y_col = 'label',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed = 726,
        class_mode='categorical')

#train_generator.class_indices


validation_generator = data_generator.flow_from_dataframe(
        dataframe=df_val,
        x_col = 'filename',
        y_col = 'label',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=726,
        class_mode='categorical')

validation_generator.class_indices

num_classes =len(train_generator.class_indices)


### Creat model
def get_compiled_model():
    if not args.model_name in ['IRV2', 'ResNet50', 'DenseNet121', 'InceptionV3']:
        raise Exception('Pre-trained network not exists. Please choose IRV2/ResNet50/DenseNet121/InceptionV3 instead')
    else:
        if args.model_name == 'IRV2':
            if database == 'RadImageNet':
                model_dir ="../RadImageNet_models/RadImageNet-IRV2-notop.h5"
                base_model = InceptionResNetV2(weights=model_dir, input_shape=(image_size, image_size, 3), include_top=False,pooling='avg')
            else:
                base_model = InceptionResNetV2(weights='imagenet', input_shape=(image_size, image_size, 3),include_top=False,pooling='avg')
        if args.model_name == 'ResNet50':
            if database == 'RadImageNet':
                model_dir = "../RadImageNet_models/RadImageNet-ResNet50-notop.h5"
                base_model = ResNet50(weights=model_dir, input_shape=(image_size, image_size, 3), include_top=False,pooling='avg')
            else:
                base_model = ResNet50(weights='imagenet', input_shape=(image_size, image_size, 3), include_top=False,pooling='avg')
        if args.model_name == 'DenseNet121':
            if database == 'RadImageNet':
                model_dir = "../RadImageNet_models/RadImageNet-DenseNet121-notop.h5"
                base_model = DenseNet121(weights=model_dir, input_shape=(image_size, image_size, 3), include_top=False,pooling='avg')
            else:
                base_model = DenseNet121(weights='imagenet', input_shape=(image_size, image_size, 3), include_top=False,pooling='avg')
        if args.model_name == 'InceptionV3':
            if database == 'RadImageNet':
                model_dir = "../RadImageNet_models/RadImageNet-InceptionV3-notop.h5"
                base_model = InceptionV3(weights=model_dir, input_shape=(image_size, image_size, 3), include_top=False,pooling='avg') 
            else:
                base_model = InceptionV3(weights='imagenet', input_shape=(image_size, image_size, 3), include_top=False,pooling='avg')
    if args.structure == 'freezeall':
        for layer in base_model.layers:
            layer.trainable = False
    if args.structure == 'unfreezeall':
        pass
    if args.structure == 'unfreezetop10':
        for layer in base_model.layers[:-10]:
            layer.trainable = False
    y = base_model.output
    y = Dropout(0.5)(y)
    predictions = Dense(num_classes, activation='softmax')(y)
    model = Model(inputs=base_model.input, outputs=predictions)
    adam = Adam(lr=args.lr)
    model.compile(optimizer=adam, loss=BinaryCrossentropy(), metrics=['acc'])
    return model



# Open a strategy scope.
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    model = get_compiled_model()




filepath="models/pneumonia-"+args.structure+ "-" + database + "-" + args.model_name + "-" + str(image_size) + "-" + str(batch_size) + "-"+str(args.lr)+ ".h5"  
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))



train_steps =  len(train_generator.labels)/ batch_size
val_steps = len(validation_generator.labels) / batch_size




history = model.fit_generator(
        train_generator,
        epochs=num_epoches,
        steps_per_epoch=train_steps,
        validation_data=validation_generator,
        validation_steps=val_steps,
        use_multiprocessing=True,
        workers=15,
        callbacks=[tensorboard,checkpoint])

### Save training loss
train_loss = list(history.history['loss'])
train_acc = list(history.history['accuracy'])
val_loss = list(history.history['val_loss'])
val_acc = list(history.history['val_accuracy'])
df_loss = pd.DataFrame({'train_loss':train_loss, 'train_acc':train_acc, 'val_loss':val_loss, 'val_acc':val_acc})



d_loss.to_csv("loss/pneumonia-"+args.structure+ "-" + database + "-" + args.model_name + "-" + str(image_size) + "-" + str(batch_size) + "-"+str(args.lr)+ ".csv", index=False)





