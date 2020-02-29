#import all necessary packages
from __future__ import print_function
from keras.models import load_model
import argparse, cv2
import numpy as np
from imutils import paths

#define image to feature vector converter
def image_to_feature_vector(image, size = (32, 32)):
    #return flat images
    return cv2.resize(image, size).flatten()

#construct the argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True, help="path to trined model")
ap.add_argument("-t","--test_images", required = True, help = "path to test images")
ap.add_argument("-b", "--batch_size", type = int, default = 32, help= "size of the mini batches passed to network")
args = vars(ap.parse_args())

#initialize the class labels as dog and cat
CLASSES = ["cat", "dog"]

#load the model
model = load_model(args["model"])

#loop over testing images
for imagePath in paths.list_images(args["test_images"]):
    #load the image and resize it to fixed 32 * 32 size then extract its feature
    print("[INFO] classifying {}".format(imagePath[imagePath.rfind("/") + 1 :]))
    image = cv2.imread(imagePath)
    features = image_to_feature_vector(image) / 255.0
    features = np.array([features])

    #predict the extracted features and pre-train neural network
    probs = model.predict(features)[0]
    prediction = probs.argmax(axis = 0)

    #draw the class and prob on the image
    label = "{}:{:.2f}%".format(CLASSES[prediction], probs[prediction] * 100)
    cv2.putText(image, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
