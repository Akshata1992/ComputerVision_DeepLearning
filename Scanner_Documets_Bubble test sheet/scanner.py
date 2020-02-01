#Import all neccessary packages
from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

#construct the argument parser and parse the arguments
ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to image")
args=vars(ap.parse_args())

#load the image and compute the old height to the new heigth,clone it and resize it
image=cv2.imread(args["image"])
ratio=image.shape[0]/500.0
origin = image.copy()
image=imutils.resize(image,height = 500)

# Convert the image to gray scale,blur it and find edges
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(image,(5,5),0)
edged = cv2.Canny(image,75,200)

# show the original and edged images
print("Detecting the edges..")
cv2.imshow("Original",image)
cv2.imshow("Edged",edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

#find the contours in edged image,keeping only the large one and initialize the screen contour
cnts = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts,key = cv2.contourArea,reverse=True)[:5]

#loop over the contours

for c in cnts:
    #approximate the contour
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,0.02*peri,True)

    #if our approximated contour has four points then we have found our screen
    if len(approx) == 4:
        screencnt = approx
        break

print("Find the contours on paper...")
cv2.drawContours(image,[screencnt],-1,(0,255,0),2)
cv2.imshow("Outliner",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#apply the fourier-transform for top-bottom view of the original image
warped =four_point_transform(origin, screencnt.reshape(4,2)*ratio)

#convert the warped image to gray and threshold it
warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
T = threshold_local(warped,11,offset = 10,method = 'gaussian')
warped = (warped > T).astype("uint8") * 255

#show the image
print("Scanned Image...")
cv2.imshow("Original", imutils.resize(origin, height=650))
cv2.imshow("Scanned", imutils.resize(warped,height =650))
cv2.waitKey(0)
