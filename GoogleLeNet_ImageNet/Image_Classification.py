#import all necessary packages
import numpy as np
import argparse, cv2, time

#construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required = True, help = "path to the input image")
ap.add_argument("-p","--prototxt", required = True, help = "path to caffe model")
ap.add_argument("-m", "--model", required= True, help = "path the trained model")
ap.add_argument("-l", "--label", required = True, help = "path to the ImageNet labels")
args = vars(ap.parse_args())

#load the input image from disk
image = cv2.imread(args["image"])

#load the class label from disk
rows = open(args["label"]).read().strip().split("\n")
#prepare comprehension list by searching space and delimiter ","
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

#CNN requires fixed dimentions of input image. To ensure that image should be normalised
blob = cv2.dnn.blobFromImage(image, 1, (224,224), (104, 117, 123))

#load the model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#set the input
net.setInput(blob)
preds = net.forward()

#sort the indexes of probabilities in descending order
idxs = np.argsort(preds[0])[::-1][:5]

#loop over top 5 predictions and display them
for (i, idx) in enumerate(idxs):
    #draw the top prediction on it
    if i == 0:
        text = "Label: {}, {:.2f}%".format(classes[idx],preds[0][idx] * 100)
        cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
	# display the predicted label + associated probability to the
	# console
    print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,classes[idx], preds[0][idx]))
# display the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
