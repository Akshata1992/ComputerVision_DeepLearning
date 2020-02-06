#import all necessary packges
import numpy as np
import argparse
import imutils
import dlib
import cv2
from imutils import face_utils

#construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-p","--shape_predictor", required=True,help="path to facial landmark oredictor")
ap.add_argument("-i","--image", required=True,help="path to the image")
args=vars(ap.parse_args())

# intialise the dlib  face detector and facial mark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#load the image,resize it and convert it to grayscale image
image= cv2.imread(args["image"])
image= imutils.resize(image,width=500)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#detect the  face in the grayscale image
rects = detector(gray,1) #(source,upscaling the resolution)

#loop over the face detections
for (i, rect) in enumerate(rects):
    #determine the facial landmarks for the face region
    #then convert the facial landmark (x,y) coordinates to a Numpy array
    shape=predictor(gray,rect)
    shape=face_utils.shape_to_np(shape)

    #convert dlib rectangle to opencv bounding box (x,y,w,h) then draw the face bounding box
    (x,y,w,h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

    #show the face number
    cv2.putText(image,"Face#{}".format(i+1),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    #Loop over (x,y) coordinates for the facial landmarks and draw them on the image
    for (x,y) in shape:
        cv2.circle(image,(x,y),1,(0,0,255),-1)

#show the output image with the face detections + facial landmarks
    cv2.imshow("output",image)
    cv2.waitKey(0)
