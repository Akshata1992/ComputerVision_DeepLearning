#import all necessary packages
import numpy as np
import imutils
from imutils import contours
from transform import four_point_transform
import argparse
import cv2

#construct the argument parser and pass the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to the bubble test image")
args= vars(ap.parse_args())

#define the anwer keys such that question number maps to answer key
answer_key = {0:1,1:4,2:0,3:3,4:1}

#Lets pre process imput image
image= cv2.imread(args["image"])
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(5,5),0)
edged = cv2.Canny(blurred,0.75,200)

cv2.imshow("Edged",edged)
cv2.waitKey(0)

#find contours in edged image then initiate the contours correspond to the document
cnts = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
dcoCnts = None

#ensure that atleast one contour was found
if len(cnts) > 0:
    #sort the contours according to size, descending order
    cnts = sorted(cnts,key = cv2.contourArea,reverse = True)

    #loop over the contour to approximate
    for c in cnts:
        # compute the peri
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.02*peri,True)
        #if our approx contour has four points then we found our paper
        if len(approx) == 4:
            docCnts = approx
            break

# apply the four point transform to turn the original and grayscale image top-bottom view
paper = four_point_transform(image, docCnts.reshape(4,2))
warped = four_point_transform(gray,docCnts.reshape(4,2))

#apply the threshold on warped image to identify about grading on the paper
thresh = cv2.threshold(warped,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]

#once again apply contour finding techniques to find the bubbles
cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questioncnts=[]

#loop over the contours
for c in cnts:
    # compute the bounding box of the contour then use the contour to derive the aspect ratio
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w/ float(h)

    #in order to lable the contour as question
    #region should be sufficiantely wide,sufficiantely tall and aspect rratio should be = 1
    if w >= 20 and h >= 20 and ar >=0.9 and ar <= 1.1:
        questioncnts.append(c)

# sort the question contours, to-bottom then initialize the correct answer
questioncnts = contours.sort_contours(questioncnts,method = "top-to-bottom")[0]
correct= 0

#each questions has 5 possible answer,loop over each questions in batched of 5
for (q,i) in enumerate(np.arange(0,len(questioncnts),5)):
    #sort the contour for the current question fromleft to right then initialize the bubbled answer
    cnts = contours.sort_contours(questioncnts[i:i+5])[0]
    bubbled=None
    flag = False
    #loop over the sorted contour
    for (j,c) in enumerate(cnts):
        #construct the mask that will reveal only current bubble of the question
        mask = np.zeros(thresh.shape,dtype="uint8")
        cv2.drawContours(mask,[c],-1,255,-1)

        #apply the mask to thresholded image then count the no of non zero pixel in the bubble area
        mask = cv2.bitwise_and(thresh,thresh,mask=mask)
        total = cv2.countNonZero(mask)

        # if the current total has large number of non-zero pixel, then we are examining the correctly answered bubble
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)

    # initialize the contour color and index of the correct answer
    color = (0,0,255)
    k = answer_key[q]

    # check to see if bubbled answer is correct
    if k==bubbled[1]:
        color = (0,255,0)
        correct +=1


    cv2.drawContours(paper,[cnts[k]],-1,color,2)

    # draw the outline of the correct answer on the test


# grab the test take
score = (correct/5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper,"{:.2f}%".format(score),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
cv2.imshow("Original",image)
cv2.imshow("Exam",paper)
cv2.waitKey(0)
