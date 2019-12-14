
# load and display an image with Matplotlib
from matplotlib import image
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras import optimizers
from mnist import MNIST
from keras.models import load_model
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
# load image as pixel array
# img = Image.open('Training/000006.jpg')
# rsize = img.resize((np.array(img.size)/10).astype(int))
# rsize.save('Training/resized_image_000006.jpg')


# display the array of pixels as an image


def resize_image():
    path = 'Training/'
    df = pd.read_csv("train_files.csv")

    for files in df.file_name:
        img = Image.open(path+files)
        img = img.resize((np.array(img.size)/5).astype(int))
        img.save(path+'resize_image'+files)


# resize_image()


def one_hot_encode(tags):
    # create empty vector
    encoding = np.zeros(5, dtype='uint8')
    # mark 1 for each tag in the vector
    for tag in tags:
        encoding[tag] = 1
    return encoding


path = 'Training/'
df = pd.read_csv("train_files.csv")
x_train, y_train = list(), list()
label = list()

for files in df.file_name:
    photo = image.imread(path+'resize_image'+files)
    x_train.append(photo)
    labels = df.annotation[df.file_name == files]
    y_train.append(one_hot_encode(labels))
    label.append(labels)
    # y_train.append((labels))

x_train = np.asarray(x_train)
label = df.annotation
label = np.asarray(label)
label = np.transpose(label)
# x_train = x_train.reshape(x_train.shape[0], 50, 50, 3)
batch_size = 16
epochs = 5
lrate = 0.0001
dpout = 0.3
no_hidd_neurons = 256

hyp_param = {'batch_size': batch_size, 'epochs': epochs, 'no_neuron': no_hidd_neurons,
             'lrate': lrate, 'dpout': dpout, 'optimizer': 'adam', 'act_func': 'softmax', 'hidden_act': 'relu'}


def resnet(my_dict):
    num_classes = 5
    model = ResNet50(weights='imagenet',
                     include_top=False,
                     input_shape=(96, 128, 3))

    x = model.output
    x = GlobalAveragePooling2D()(x)
    # we add dense layers so that the model can learn more complex functions and classify for better results.
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    # x = Dense(512, activation='relu')(x)  # dense layer 2
    # x = Dense(256, activation='relu')(x)  # dense layer 3
    preds = Dense(num_classes, activation='softmax')(x)  # final layer with softmax activation
    model = Model(inputs=model.input, outputs=preds)
    adam = optimizers.adam(lr=my_dict['lrate'])
    model.compile(optimizer=my_dict['optimizer'],
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def create_model(my_dict):
    # Importing the required Keras modules containing model and layers
    # Creating a Sequential Model and adding the layers
    num_classes = 5
    input_shape = (96, 128, 3)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    print('Add first conv networks')
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print('Added first max pooled')
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    print('Add second conv networks')
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print('Added second max pooled')
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    print('Add third conv networks')
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print('Added third max pooled')
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers so that tf.nn.relu
    model.add(Dense(my_dict['no_neuron'], activation=my_dict['hidden_act']))
    model.add(Dropout(my_dict['dpout']))  # fraction of input layers to drop
    # model.add(Dense(128, activation=my_dict['hidden_act']))
    # model.add(Dropout(my_dict['dpout']))  # fraction of input layers to drop
    model.add(Dense(num_classes, activation=my_dict['act_func']))
    adam = optimizers.adam(lr=my_dict['lrate'])
    model.compile(optimizer=my_dict['optimizer'],
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# model = create_model(hyp_param)


model = resnet(hyp_param)
history = model.fit(x=x_train, y=label,
                    epochs=hyp_param['epochs'], batch_size=hyp_param['batch_size'])
model.save('new_Batch Size = '+str(hyp_param['batch_size']) +
           'learning_rate = '+str(hyp_param['lrate'])+'trained_model.h5')
