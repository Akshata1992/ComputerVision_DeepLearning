#build smaller version of VGGNet convolution Neural Network
#import all necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras import backend as K

#define smallVGGNet class
class SmallerVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        #initialize the model and input shape and channel dimentions
        model = Sequential()
        inputShape = (height,width,depth)
        chanDim = -1

        #if we are using channels first then update the input dimentions and channel dimentions
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        #build the layers for our model:
        #CONV => RELU => POOL
        model.add(Conv2D(32,(3,3),input_shape = inputShape,padding='same'))
        model.add(Activation('relu'))
        # Note: After every activation, batch normalization is used to maintain the weights
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

        #construct the next layers
        # CONV => RELU * 2 => POOL
        #Note: Always increase the filter size of convolution and reduce the pool size,not too much!,to get richer set of features
        model.add(Conv2D(64,(3,3),input_shape=inputShape,padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64,(3,3),input_shape=inputShape,padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        #construct the next set of layers
        # CONV => RELU * 2 => POOL
        model.add(Conv2D(128,(3,3),input_shape = inputShape, padding = 'same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(128,(3,3),input_shape = inputShape, padding = 'same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        #Construct the final fully connected layer with softmax classification
        #FC => RELU and softmax classifier
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.5))

        #Softmax classifier
        model.add(Dense(classes))
        #note: Softmax activation function gives each transformed logit/sum of transformed logits whose sum is equal to prob of 1
        model.add(Activation('softmax'))

        #return the model
        return model
