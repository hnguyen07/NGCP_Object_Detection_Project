import time
import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

#piicodev libraries for sensor
from PiicoDev_VL53L1X import PiicoDev_VL53L1X
from time import sleep
import utils


def run() -> None:
	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
	
	#CHANGE NAME OF FILE YOU'RE USING HERE AND SET USE_CORAL TO TRUE IF USING EDGETPU LINE 22
	#CHANGE MAX RESULTS LINE 23 TO DETECT MORE OBJECTS AT SAME TIME
	base_options = core.BaseOptions(file_name="android_edgetpu.tflite", use_coral=True, num_threads=4)
	detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.3)
	options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
	detector = vision.ObjectDetector.create_from_options(options)
	
	#declare visualization parameters
	row_size = 20  # pixels
	left_margin = 24  # pixels
	right_margin = 300
	text_color = (0, 0, 255)  # red
	font_size = 1
	font_thickness = 1
	fps_avg_frame_count = 10
	counter, fps = 0, 0
	start_time = time.time()
	
	distSensor = PiicoDev_VL53L1X()
	while cap.isOpened():
		success, image = cap.read()
		counter += 1
		image = cv2.flip(image, 1)

		# Convert the image from BGR to RGB as required by the TFLite model.
		rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# Create a TensorImage object from the RGB image.
		input_tensor = vision.TensorImage.create_from_array(rgb_image)

		# Run object detection estimation using the model.
		detection_result = detector.detect(input_tensor)

		# Draw keypoints and edges on input image
		image = utils.visualize(image, detection_result)
		
		#sensor
		#If multiple snesors plugged, it choose shortest distance
		dist = distSensor.read()  # read the distance in millimetres
		distance_text = 'Distance = {:.1f}'.format(dist) + 'mm'
		text_location = (left_margin, row_size)
		cv2.putText(image, distance_text, text_location, cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)
		#sleep(0.1)

		#FPS calculation
		if counter % fps_avg_frame_count == 0:
			end_time = time.time()
			fps = fps_avg_frame_count / (end_time - start_time)
			start_time = time.time()

		#Show FPS
		fps_text = 'FPS = {:.1f}'.format(fps)
		text_location = (right_margin, row_size)
		cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)
		
		#Stop the program if ESC key is pressed
		if cv2.waitKey(1) == 27:
			break
		cv2.imshow('object_detector', image)

	cap.release()
	cv2.destroyAllWindows()

run()