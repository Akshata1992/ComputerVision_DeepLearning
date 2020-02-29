from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
import argparse, cv2, os

#define a function to accept image and resize it w/o aspect ratio
def image_to_feature_vector(image,size = (32,32)):
    #Resize the image to a fixed size and then flatten the image to raw pixel intensities
    return cv2.resize(image,size).flatten()

##construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to data set")
ap.add_argument("-m","--model",required=True,help="path to save the model")
args = vars(ap.parse_args())

#grab the list of images
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))

#initialize the data matrix and label matrix
data = []
labels = []

#Loop over input dataset
for (i,imagePath) in enumerate(imagePaths):
    #load the image and extract the class label
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    #convert the image to flat and construct the data and label lists
    features = image_to_feature_vector(image)
    data.append(features)
    labels.append(label)

    #show an update every 1000 image
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))

le= LabelEncoder()
labels = le.fit_transform(labels) # convert all string to interger

#scale the input image to range [0,1]
#transform the labels into one hot vector [0,num_classes]
data = np.array(data)/ 255.0
labels = np_utils.to_categorical(labels,2)

#partition the data inot training and test split
(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.25, random_state=42)

#define the model
model = Sequential()
model.add(Dense(768, input_dim = 3072, init = "uniform", activation="relu"))
model.add(Dense(384, activation="relu", kernel_initializer="uniform"))
model.add(Dense(2))
model.add(Activation("softmax"))

#train the model_selection
print("[INFO] compiling the model....")
sgd = SGD(lr = 0.01)
model.compile(loss = "binary_crossentropy", optimizer = sgd, metrics = ["accuracy"])
model.fit(trainData, trainLabels, epochs = 50, batch_size = 128, verbose = 1)

#show the accuracy on the testing set
print("[INFO] evaluating the test data...")
(loss,accuracy) = model.evaluate(testData, testLabels, batch_size= 128, verbose =1)
print("[INFO] loss = {:.4f}, accuracy = {:.4f}%".format(loss, accuracy * 100))

#save the model
print("[INFO] saving the model...")
model.save(args["model"])
