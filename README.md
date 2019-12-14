Motivation
============

Weather stress and climate change are major challenges for agriculture. Drought tolerant soybean varieties are needed but breeding for drought tolerance in soybean, is difficult. Soybean breeders evaluate thousands of field plots annually in order to select the best plants to advance for further testing or to release as a new variety. The most widely used indicator of drought tolerance/sensitivity in soybean is leaf wilting during periods of water stress1, and breeders collect this data in the most low-tech way imaginable — walking the field with a notebook and writing down ratings to indicate how wilted each field plot looks.

## Contents

- [Using Transfer Learning Models like ResNet](README.md#Using-Transfer-Learning-Models-like-ResNet)
- [CNN Network from Scratch](README.md#CNN-Network-from-Scratch)

## Using Transfer Learning Models like ResNet

We chose to examine different approaches like transfer learning model incorporation (Res-NET50), building a customized CNN and train it specific for the data set. Res-NET50 is a pre-trained convolutional neural network that is trained with millions of images from the Image-NET data set. The network is 50 Convolution layers and can classify 1000 unique objects. Designers can add their customized fully connected network to extract the features they want. Other transfer learning models like VGG19 was examined as well. VGG19 has 19 Convolution Layers and Inception has 22 convolution layers. Increasing the depth of the model does not assure good accuracy because of the vanishing gradients problem with deeper networks. ResNet-50 overcomes this
vanishing gradient problem with the help of a technique called skip connection. Training a model from scratch might not be
a good approach with small datasets like these. We import ResNet50 pre-trained model and remove the fully connected network. On top of the ResNet50 Convolution layer, we add our first layer - global average pooling which gives us the average pixel density of each feature. We then added 2 fully connected layers, with 128 and 5 neurons respectively. The last layer is the output layer with 5 classes for 5 different one-hot encoded output vectors.

## CNN Network from Scratch
we chose to build a customized CNN+Fully
connected Neural Network and train it specific for the data
set. In this methodology, we implemented a CNN based model from
scratch. We have 5 convolution layers with batch normalization
added for a smooth learning process. On top of these
convolution layers, we have the fully connect neural network
with one hidden layer and output layer

![Model Summary](Screenshot-2019-12-02-at-6.04.21-PM.png)
  
