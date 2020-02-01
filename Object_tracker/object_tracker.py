#import all neccessary packages
import numpy as np
import cv2,imutils,time,argparse
from imutils.video import VideoStream
from collections import deque # preferred over list for quicker append and pop operations. To record the track of ball.

#construct the argparse to parse the arguments
ap= argparse.ArgumentParser()
ap.add_argument("-v","--video",help="video stream to track the object")
ap.add_argument("-b","--buffer",type=int,default=64,help= "max buffer size to determine the size of deque")
args=vars(ap.parse_args())

#define the upper and lower boundaries of the green ball in the HSV color space, then
# initialize the list of tracks
greenlower= (29,86,6)
greenupper= (64,255,255)
pts = deque(maxlen=args["buffer"])

#if video path was not supplied grab the reference to the webcam

if not args.get("video",False):
    vs=VideoStream(src=0).start()
else:
    #grab the reference to video file
    vs = cv2.VideoCapture(args["video"])

#allow the video file or webcam to warm up
time.sleep(2.0)

#Keep looking

while True:
    frame=vs.read() # returns 2 - tuple

    #handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame
    # if we are viewing the video and not grabbed any frame that means
    # we have reached end of video
    if frame is None:
        break

    # resize the frame, blurr it then convert the image to HSV
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame,(11,11),0)
    hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)

    #construct a mask for the color "green", then perform
    #series of dilation and erosion to remove any small blob in the mask
    mask = cv2.inRange(hsv,greenlower,greenupper)
    mask = cv2.erode(mask,None,iterations=2)
    mask = cv2.dilate(mask,None,iterations=2)

    # find contour in the mask and initialize the (x,y) center of the ball
    cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    #only proceed if atleast one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask then use it to compute min enclosing circle and centroid
        c = max(cnts,key=cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"]/ M["m00"]), int(M["m01"]/ M["m00"]))

        #only proceed if radius meets min size
        if radius > 10:
            #draw the circle and centroid of the frame
            #then update the list of tracked points
            cv2.circle(frame, (int(x),int(y)),int(radius),(0,255,255),2)
            cv2.circle(frame,center,5,(0,0,255),-1)

    #update the pnts in deque
    pts.appendleft(center)

    #draw the contrail of the ball by passing the pass (x,y) coordinates of the ball that is been detected
    #loop over the set of tracked points
    for i in range(1, len(pts)):
        #if either of the tracked points are none,ignore them
        if pts[i-1] is None or pts[i] is None:
            continue

        #otherwise compute the thickness of the line then draw the connecting lines
        thickness = int(np.sqrt(args["buffer"]/ float(i+1))* 2.5)
        cv2.line(frame, pts[i-1],pts[i],(0,0,255),thickness)

    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

    #if key 'q' is pressed, stop the video
    if key == ord('q'):
        break

# if we are not  using a vedio file, them stop the video stream
if not args.get("video", False):
    vs.stop()
#otherwise release the camera
else:
    vs.release()

#close all the widows
cv2.destroyAllWindows()
