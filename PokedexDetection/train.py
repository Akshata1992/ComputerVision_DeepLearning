# Usage -- Implementing our CNN + Keras training script
#set the matplotlib to backend so that images can be saved in the backend
import matplotlib
matplotlib.use("Agg")

#Import all necessary packages
from keras.preprocessing.image import ImageDataGenerator # class is used for data augmentation(apply random transformations)
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
#Note: Transforms our class labels to one hot encoder
from sklearn.model_selection import train_test_split
from smallerVGGNet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import pickle
import os, cv2, random

#Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required =True, help = "path to the dataset")
ap.add_argument("-m", "--model", required=True,help="path to the output model")
ap.add_argument("-l","--labelbin",required=True,help="path to the label binarization")
ap.add_argument("-p","--plot",type=str,default="plot.png",help="path to the output accuracy/loss plot")
args=vars(ap.parse_args())

#initialize epochs,learning rate,batch_size and image dimentions
EPOCHS = 50
INIT_LR = 1e-3
BS = 32
IMAGE_DIM = (96, 96, 3)

#initialize data and labels
data = []
labels = []

#grab the image paths and randomly shuffle them
print("[INFO] loading images....")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

#loop over the input images
for imagePath in imagePaths:
    #load the image,pre-process it and add to the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIM[1], IMAGE_DIM[0]))
    image = img_to_array(image)
    data.append(image)

    #extract the class labels
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

#scale the raw pixel intensitites to the range 255
data = np.array(data,dtype = "float")/ 255.0
labels = np.array(labels)
print("[INFO] data matrix {:.2f}MB".format(data.nbytes/(1024*1000.0)))

#binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

#split the data inot training,testing with 50% transformations
(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.2,random_state=42)

#construct the data augmentation for images
aug = ImageDataGenerator(rotation_range = 25,width_shift_range = 0.1,height_shift_range = 0.1,shear_range = 0.2,zoom_range = 0.2,
                             horizontal_flip=True,fill_mode = "nearest")

#initialize the model
print("[INFO] compiling the model....")
model = SmallerVGGNet.build(width=IMAGE_DIM[1],height = IMAGE_DIM[0],depth=IMAGE_DIM[2],classes=len(lb.classes_))
opt = Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss="categorical_crossentropy",optimizer=opt, metrics=["accuracy"])

#fit the model
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX,trainY,batch_size=BS), validation_data=(testX,testY),steps_per_epoch =len(trainX)// BS,epochs=EPOCHS,verbose=1)

#save the model
model.save(args["model"])

#save the label binarizer to disk
print("[INFO] serializing the label binarizer....")
f= open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()

#plot the training and loss function
plt.style.use("ggplot")
plt.figure()
N=EPOCHS
plt.plot(np.arange(0,N),H.history["loss"],label="train_loss")
plt.plot(np.arange(0,N),H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,N),H.history["acc"],label="train_acc")
plt.plot(np.arange(0,N),H.history["val_acc"],label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])

#convert class labels to list
lb_ = pickle.loads(open(args["labelbin"], "rb").read())
class_labels = lb_.classes_.tolist()
print("[INFO] class labels : {}".format(class_labels))

#convert the model from Keras to TensorFlow using following code snippet
from tensorflow import lite
converter = lite.TFLiteConverter.from_keras_model_file( 'pokedex.hdf5')
tfmodel = converter.convert()
open ("pokedex.tflite" , "wb") .write(tfmodel)
