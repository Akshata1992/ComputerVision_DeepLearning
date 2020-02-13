#import all necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread #play alarm in separate thread without pausing the script
import numpy as np
import imutils, time, cv2, dlib, argparse, playsound

#define sound alarm function to detect the audio file
def sound_alarm(path):
    playsound.playsound(path)
    return

def eye_aspect_ratio(eye):
    #compute the euclidean distance between 2 sets of vertical coordinates(x,y-coordinates)
    A = dist.euclidean(eye[1],eye[5])
    B = dist.euclidean(eye[2],eye[4])
    #compute the euclidean distance between horizontal pair
    C = dist.euclidean(eye[0],eye[3])

    #Compute the eye aspect ratio
    ear = (A + B)/(2.0 * C)
    return ear

#construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-p","--shape_predictor", required=True,help="path to shape predictor algm")
ap.add_argument("-a","--alarm",type=str,default="",help="path to .WAV file")
ap.add_argument("-w","--webcam", type=int,default=0,help="webcam for video stream")
args=vars(ap.parse_args())

#define two constants,one for eye aspect ratio to indicate blink
#and another for number of consecutive frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

#initialise the counter and total no of frames
COUNTER = 0
ALARM_ON = False

#initialise the dlib face detector and create facial landmarks predictors
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#grab the indexes of facial marks of left eye and right eye
(lstart,lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart,rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#start the  live stream or webcam
vs= VideoStream(src=args["webcam"]).start()
#allow the video file or web cam to warm up
time.sleep(1.0)

#loop over the frames from video stream
while True:

    #grab the frames video stream,thread it,resize it and convert it to gray scale
    frame = vs.read()
    frame = imutils.resize(frame, width = 500)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detect the face in grayscale image
    rects = detector(gray,0)

    #loop over facial landmark detections
    for rect in rects:
        #determine the facial landmark and convert the coordinates to Numpy array
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)

        # extract the coordinates for both left and right eye
        #compute the eye aspect ratio for boththe eyes
        leftEye = shape[lstart:lend]
        rightEye = shape[rstart:rend]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        #average the eye aspect ratio for better blink estimate
        #considering both the eyes blinkat same time
        ear = (leftEAR + rightEAR)/ 2.0

        #compute the convex hull of left eye and right eye then visualize each of the eye
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull],-1,(0,255,0),1)
        cv2.drawContours(frame,[rightEyeHull],-1,(0,255,0),1)

        #check to see if EAR is below to threshold to detect the blink
        #if so, icrement the frame counter
        if ear < EYE_AR_THRESH:
            COUNTER +=1

            #if eye were closed for sufficiant number of frames
            # then sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:

                if not ALARM_ON:
                    ALARM_ON = True

                    #check to see if alarm file was supplied,
                    #and if so,start the thread to play the
                    #sound in the background
                    if args["alarm"] != "":
                        t = Thread(target = sound_alarm,args=(args["alarm"],))
                        t.deamon = True
                        t.start()

                #draw the text on the frame for drowsiness
                cv2.putText(frame,"DROWSINESS ALERT",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        else:
            COUNTER = 0
            ALARM_ON = False

        #draw the text on the frame along with EAR
        cv2.putText(frame,"EAR:{:.2f}".format(ear),(300,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    #show the image
    cv2.imshow("frame",frame)
    key = cv2.waitKey(1) & 0xFF

    # check if any key is pressed
    if key == ord('q'):
        break

#clean up
cv2.destroyAllWindows()
vs.stop()
