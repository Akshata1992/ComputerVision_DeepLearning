#Eye blink detection with OpenCv,dlib on Python
#import all necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import time
import cv2
import dlib
import imutils

def eye_aspect_ratio(eye):
    #compute the euclidean distance between 2 sets of vertical coordinates(x,y-coordinates)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    #compute the euclidean distance between horizontal pair
    C = dist.euclidean(eye[0], eye[3])

    #Compute the eye aspect ratio
    ear = (A + B)/ (2.0 * C)

    #return EAR
    return ear

#construct the argument parser and parse the argument

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape_predictor", required=True, help="model for face and facial detector")
ap.add_argument("-v","--video",type=str,default="",help="path to input video file")
args=vars(ap.parse_args())

#define two constants,one for eye aspect ratio to indicate blink
#and another for number of consecutive frames the eye must be below the threshold
EYE_AR_THRESH = np.arange(0.28,0.32)
EYE_AR_CONSEC_FRAMES = 3

#initialise the counter and total no of frames
COUNTER= 0
TOTAL = 0 # no of blinks

#initialise the dlib face detector and create facial landmarks predictors
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#grab the indexes of facial marks of left eye and right eye
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#check for the input source a video file or live stream or webcam
#start the video stream thread or video file
if not args.get("video",False):
    vs= VideoStream(src=0).start()
    fileStream = False
else:
    vs= FileVideoStream(args["video"]).start()
    fileStream = True
#allow the video file or web cam to warm up
time.sleep(1.0)
print(vs)
#loop over the frames from video stream
while True:
    #if file video stream then check for if any frames left to buffer
    if fileStream and not vs.more():
        break

    #grab the frames video stream,thread it,resize it and convert it to gray scale
    frame= vs.read()
    frame= imutils.resize(frame, width = 700)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detect the face in grayscale image
    rects = detector(gray,0)

    #loop over facial landmark detections
    for rect in rects:
        #determine the facial landmark and convert the coordinates to Numpy array
        shape= predictor(gray,rect)
        shape= face_utils.shape_to_np(shape)

        # extract the coordinates for both left and right eye
        #compute the eye aspect ratio for boththe eyes
        leftEye = shape[lstart:lend] #left eye
        rightEye = shape[rstart:rend] # right eye
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        #average the eye aspect ratio for better blink estimate
        #considering both the eyes blinkat same time
        ear = (leftEAR + rightEAR) / 2.0

        #compute the convex hull of left eye and right eye then visualize each of the eye
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame,[leftEyeHull],-1,(0,255,0),1)
        cv2.drawContours(frame,[rightEyeHull],-1,(0,255,0),1)

        #check to see if EAR is below to threshold to detect the blink
        #if so, icrement the frame counter
        if ear < EYE_AR_THRESH:
            COUNTER +=1

        else:
            #if eye were closed for suffieciant number,increment the total counter
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL +=1

            #reset the eye frame counter
            COUNTER = 0

        #draw the total number of blinks on the frame along with EAR
        cv2.putText(frame,"Blinks: {}".format(TOTAL),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,"EAR: {:.2f}".format(ear),(300,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    #show the frame
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

    #check for wait key pressed
    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
