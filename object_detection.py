import cv2 as cv
import numpy as np
import time
import os
import datetime
import sys
import subprocess

# load yolov3 weights and config files into network
# Weight file: it’s the trained model, the core of the algorithm to detect the objects.
# Cfg file: it’s the configuration file, where there are all the settings of the algorithm.
# Name files: contains the name of the objects that the algorithm can detect.
net = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
i=0
while(i<5):
	# read the absolute path
	script_dir = os.path.dirname(__file__)
	# call the .sh to capture the image
	os.system('./takeimage.sh')
	#get the date and time, set the date and time as a filename.
	currentdate = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
	# create the relative path
	rel_path = currentdate +".jpg"
	#  join the absolute path and created file name
	abs_file_path = os.path.join(script_dir, rel_path)
	print(abs_file_path)

	# load image/video and get dimensions
	img_path = os.path.join("photos", abs_file_path)
	print(img_path)
	img = cv.imread(img_path)  # image
	# cap = cv.VideoCapture(0)  # video input (0 is default camera/webcam)
	# while True:
	#     _, img = cap.read()
	# img = cv.resize(img, None, fx=0.4, fy=0.4)  # resize image if needed
	height, width, channels = img.shape

		# extract feature from image and resize using blob
		# three options using YOLO:
		# 320×320 it’s small so less accuracy but better speed
		# 609×609 it’s bigger so high accuracy and slow speed
		# 416×416 it’s in the middle and you get a bit of both.
	blob = cv.dnn.blobFromImage(img, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outs = net.forward(output_layers)  # results of object detection; contains objects detected, position, and confidence

		# Showing information on the screen
		# Box: contain the coordinates of the rectangle surrounding the object detected.
		# Label: it’s the name of the object detected
		# Confidence: the confidence about the detection from 0 to 1.
	class_ids = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
		    scores = detection[5:]
		    class_id = np.argmax(scores)
		    confidence = scores[class_id]
		    if confidence > 0.5:
		        # Object detected
		        center_x = int(detection[0] * width)
		        center_y = int(detection[1] * height)
		        w = int(detection[2] * width)
		        h = int(detection[3] * height)

		        # Rectangle coordinates
		        x = int(center_x - w / 2)
		        y = int(center_y - h / 2)

		        boxes.append([x, y, w, h])
		        confidences.append(float(confidence))
		        class_ids.append(class_id)
	# remove duplicate boxes for the same object
	indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

	font = cv.FONT_HERSHEY_PLAIN
	f = open("log.txt", "a")
	for i in range(len(boxes)):
		if i in indexes:
		    x, y, w, h = boxes[i]
		    label = str(classes[class_ids[i]])
		    color = colors[i]
		    cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
		    cv.putText(img, label, (x, y + 30), font, 3, color, 3)
		    f.write(time.ctime() + " " + label + "\n")
	f.close()
	#cv.imshow("Object Detection", img)
	#cv.waitKey(1)
		# key = cv.waitKey(1)
		# if key == 27:  # close window when esc key is pressed
		#     break

	cv.destroyAllWindows()
	# cap.release()
	time.sleep(1)
	i+=1

