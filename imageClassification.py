import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#preprocess images

#variables to define path of data sets
trainingDataPath = 'dataSet/trainingData'
validationDataPath = 'dataSet/validationData'

#image size
imageWidth, imageHeight = 150, 150

#number of samples propagated
batchSize = 32


trainGen = ImageDataGenerator(
     rescale = 1./255, #divide training data by 255 to rescale from [0, 255] to [0, 1]
     shear_range = 0.1,
     zoom_range = 0.2)

valGen = ImageDataGenerator(rescale = 1./255)



#training set gen
trainingGenerator = trainGen.flow_from_directory(
     trainingDataPath, 
     target_size=(imageWidth, imageHeight),     
     color_mode = "rgb",
     batch_size = batchSize,
     class_mode = 'binary')

#validation set gen
validationGenerator = valGen.flow_from_directory(
     validationDataPath,
     target_size=(imageWidth, imageHeight),
     color_mode = "rgb",
     batch_size = batchSize,
     class_mode = 'binary')



#multi layered convolutional network
model = Sequential() #sequential model to build network layer by layer
model.add(Convolution2D(32, (3, 3), input_shape=(imageWidth, imageHeight,3))) #layer to extract features from image
model.add(Activation('relu')) #apply activation function to output -- relu used for rectified linear activation // possibly use 'elu' for exponential linear activation
model.add(MaxPooling2D(pool_size=(2, 2))) #reduce spatial volume of image after convolution

model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3))) #64 convolution filters on 3rd convolution layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) #flatten data as convolution is 2D and Dense layer requires 1D data set
model.add(Dense(64)) #64 nodes to connect the neural network
model.add(Activation('relu'))
model.add(Dropout(0.5)) #set fraction rate of inputs to 0.5 at each update during training

#output layer
model.add(Dense(1))
model.add(Activation('sigmoid')) # sigmoid activation = 1 / (1 + exp(-x))


model.compile(loss='binary_crossentropy', #how accurate model is during training
              optimizer='rmsprop', #how model is updated based on the data it sees and its loss function
              metrics=['accuracy']) #monitor training and testing steps -- accuracy used for the fraction of images that are correctly classified 

EPOCHS = 30

trainFilenames = trainingGenerator.filenames
trainingSamples = len(trainFilenames)

valFilenames = validationGenerator.filenames
validationSamples = len(valFilenames)


#train model
model.fit_generator(
        trainingGenerator,
        steps_per_epoch = trainingSamples // batchSize,
        epochs = EPOCHS,
        validation_data = validationGenerator,
        validation_steps = validationSamples // batchSize,)

model.evaluate_generator(generator = validationGenerator, steps = validationSamples)


model.save('myModel.h5')
del model

model = load_model('myModel.h5')
print("Saved model")


##Extras:
#add more comments
#alter network slightly
#add optimizations
#provide reference
