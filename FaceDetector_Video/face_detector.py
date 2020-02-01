#Import all necessary packages
import numpy as np
import argparse
import cv2

#Construct the arguement parse and parse the arguemetns
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required = True, help = "path to input image")
ap.add_argument("-p","--prototxt", required = True, help = "path to caffemodel prototxt file")
ap.add_argument("-m","--model", required = True, help="path to Caffe pre-trained model")
ap.add_argument("-c","--confidence", type=float, default = 0.5,help= "minimum probability to filter weak detections")
args = vars(ap.parse_args())

#Load our serialized model from disk
print("[Info] Loading model...")
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

#Load the input image and construct the input blob from image and resizing it 300x300 pixels and normalize it
image = cv2.imread(args["image"])
(h,w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(140.0,177.0,123.0))

#pass the blob through the network and obtain the detections and predictions
print("[Info] computing object detections..")
net.setInput(blob)
detections = net.forward()

# Loop over the detections
for i in range(0, detections.shape[2]):
    #Extract the confidences(probability) associated  with the prediction
    confidence=detections[0,0,i,2]

    #Filter out the weak detections by ensuring the confidence is greaterr than min confidence
    if confidence > args['confidence']:
        #Compute the x,y coordinates of bounding box for the object
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (startX,startY,endX,endY) = box.astype('int')

        #Draw the bounding box aroung the face with associated probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startX + 10
        cv2.rectangle(image,(startX,startY),(endX,endY),(0,0,255),2)
        cv2.putText(image,text,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)

# Show the image
cv2.imshow("Output",image)
cv2.waitKey(0)
