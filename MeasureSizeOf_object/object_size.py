#import all necessary packages
from imutils import perspective
from imutils import contours
import numpy as np
import argparse,cv2,imutils
from scipy.spatial import distance as dist

#create midpoint between (x,y) cordinates
def midpoint(ptA,ptB):
    return((ptA[0]+ptB[0])*0.5,(ptA[1]+ptB[1])*0.5)

#construct the argument parser adn parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help= "path to the input image")
ap.add_argument("-w","--width",type =float,required=True,help="width of the leftmost image")
args=vars(ap.parse_args())

#load the image,grayscale it and blur it slightly
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(7,7),0)

#perform the edge detection,dilation and erosionto close the gaps inbetween the objects edges
edged = cv2.Canny(blurred,50,100)
edged = cv2.dilate(edged,None,iterations=1)
edged = cv2.erode(edged,None,iterations=1)

#find the contours in the edged map
cnts=cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#sort the contours from left to right then initialize the pixel per matrics
(cnts,_) = contours.sort_contours(cnts)
pixelsPerMetric = None

#loop over the contours individually
for c in cnts:
    #if contours is not sufficiently large,ignore it
    if cv2.contourArea(c) < 100:
        continue

    #compute the rotated bounding box of the contour
    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box,dtype="int")

    #order points in the contour such that they are arranged from top-left,
    #top-right,bottom-left and bottom-right
    box = perspective.order_points(box)
    cv2.drawContours(orig,[box.astype("int")],-1,(0,255,0),2)

    #loop over the original points and draw them
    for (x,y) in box:
        cv2.circle(orig,(int(x),int(y)),5,(0,0,255),-1)

    #unpack the ordered bounding box,then compute the mid-point tl-tr
    #co-ordinates and bl-br coordinates
    (tl,tr,br,bl) = box
    (tltrX,tltrY) = midpoint(tl, tr)
    (blbrX,blbrY) = midpoint(bl, br)
    (tlblX,tlblY) = midpoint(tl, bl)
    (trbrX,trbrY) = midpoint(tr, br)

    #draw the mid points on the image
    cv2.circle(orig,(int(tltrX),int(tltrY)),5,(255,0,0),-1)
    cv2.circle(orig,(int(blbrX),int(blbrY)),5,(255,0,0),-1)
    cv2.circle(orig,(int(tlblX),int(tlblY)),5,(255,0,0),-1)
    cv2.circle(orig,(int(trbrX),int(trbrY)),5,(255,0,0),-1)

    #draw the lines between mid points
    cv2.line(orig,(int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255,0,255),2)
    cv2.line(orig,(int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255,0,255),2)

    #compute the ecludian ditance between midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY)) #height
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY)) #width

    #if pixel per metrics is not initialized then compute
    #the ratio of pixel to supplied metrics
    if pixelsPerMetric is None:
        pixelsPerMetric = dB/args["width"]

    #compute the size of the object
    dimA = dA/pixelsPerMetric
    dimB = dB/pixelsPerMetric

    #draw the object size on the image
    cv2.putText(orig, "{:.1f}in".format(dimA),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.65,
                (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f}in".format(dimB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)

    #show the image
    cv2.imshow("Image",orig)
    cv2.waitKey(0)
