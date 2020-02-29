#Import all necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import time
import cv2
import imutils

#construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-p","--prototxt",required=True,help="path to the proto text file of model")
ap.add_argument("-m","--model",required=True,help="Path to Caffe model")
ap.add_argument("-c","--confidence",type=float,default=0.2,help="minimum probability to filter for object detection")
args = vars(ap.parse_args())

#Initialize the class list of labels to detect the object
CLASSES=["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor","glasses","mobile"]

#Generate set of bounding box of colors
COLORS = np.random.uniform(0, 255, size = (len(CLASSES),3))

#Load our serialized model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])

#Initialize the video stream and allow the camera sensor to warmup and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

#Loop over the frames from VideoStream
while True:
    #grab the frame from video and resize it to width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=1200)
    #grab the frame dimention and convert it to blob
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),0.007843,(300,300),127.5)

    #pass the blob to through network and obtain the detection
    net.setInput(blob)
    detections = net.forward()

    #loop over the detections
    for i in np.arange(0, detections.shape[2]):
        #extract the confidence of the probabilities
        confidence = detections[0,0,i,2]

        if confidence > args["confidence"]:
            #extract the index of the class label from the detections, then compute the bounding box of (x,y) coordinates
            idx = int(detections[0,0,i,1])
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX,startY,endX,endY) = box.astype("int")

            #draw the predictions on the frames
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame,(startX,startY),(endX,endY),COLORS[idx],2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame,label,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[idx],2)

    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

    if key ==ord('q'):
        break

#update the fps counter
    fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
