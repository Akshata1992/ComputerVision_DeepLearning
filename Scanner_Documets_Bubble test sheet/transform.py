#Usage - import all the neccessary packages
import numpy as np
import cv2

def order_points(pts):
    #Initialize the list of coordinates(x,y) such that entries in the list is in order of top-left,top-right,bottom-right and bottom-left
    rect = np.zeros((4,2), dtype="float32")

    #top-left will have smallest sum and bottom-right will have largest sum
    s= pts.sum(axis=1)
    rect[0]= pts[np.argmin(s)]
    rect[2]= pts[np.argmax(s)]

    #compute the difference between the points
    # the top-right point will have smallest value and bottom-left will have larger value
    diff = np.diff(pts,axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered co-ordinates
    return rect

def four_point_transform(image,pts):
    # obtain the consistent order of points and unpack them accordingly
    rect = order_points(pts)
    (tl,tr,br,bl) = rect

    #compute the width of new image which will be difference between points bootom-right to bottom-left
    #or top-right to top-left
    widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
    widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
    maxWidth = max(int(widthA),int(widthB))

    #Simillarly compute the height of new image
    heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
    heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
    maxHeight = max(int(heightA),int(heightB))

    #now that we have dimentions of new image,construct the set of destination points for top-bottom view
    #of the image, again specifying the points in order tl,tr,br,bl
    dst = np.array([
                    [0,0],
                    [maxWidth -1,0],
                    [maxWidth -1, maxHeight -1],
                    [0,maxHeight -1]], dtype="float32")

    #compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect,dst)
    warped = cv2.warpPerspective(image,M,(maxWidth,maxHeight))

    return warped
