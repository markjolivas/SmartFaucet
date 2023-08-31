import cv2
import time
import RPi.GPIO as GPIO
import numpy as np                      
import tensorflow as tf
from keras.models import load_model

solenoidHot2 = 27
solenoidCold1 = 5
solenoidCold2 = 6
solenoidHot1 = 22
motionFreeFlow = 16
motionHot = 19
motionCold = 21
motionWaterOff = 26
tempMotion = 0

GPIO.setmode(GPIO.BCM)
GPIO.setup(solenoidHot2, GPIO.OUT)
GPIO.output(solenoidHot2, False)
GPIO.output(solenoidHot2, 0)
GPIO.setup(solenoidHot1, GPIO.OUT)
GPIO.output(solenoidHot1, 0)
GPIO.setup(solenoidCold1, GPIO.OUT)
GPIO.output(solenoidCold1, 0)
GPIO.setup(solenoidCold2, GPIO.OUT)
GPIO.output(solenoidCold2, 0)
GPIO.setup(motionFreeFlow, GPIO.IN)
GPIO.input(motionFreeFlow)
GPIO.setup(motionHot, GPIO.IN)
GPIO.input(motionHot)
GPIO.setup(motionCold, GPIO.IN)
GPIO.input(motionCold)
GPIO.setup(motionWaterOff, GPIO.IN)
GPIO.input(motionWaterOff)

# 
# classNames = []
# classFile = "/home/bd-sink/Desktop/Object_Detection_Files/coco.names"
# with open(classFile, "rt") as f:
#     classNames = f.read().rstrip("\n").split("\n")
# 
# configPath = "/home/bd-sink/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
# weightsPath = "/home/bd-sink/Desktop/Object_Detection_Files/frozen_inference_graph.pb"
# 
# net = cv2.dnn_DetectionModel(weightsPath, configPath)
# net.setInputSize(320, 320)
# net.setInputScale(1.0 / 127.5)
# net.setInputMean((127.5, 127.5, 127.5))
# net.setInputSwapRB(True)
#

# Load the class labels
# labels_path = '/home/bd-sink/Desktop/Object_Detection_Files/sink-labels.txt'
# with open(labels_path, 'r') as f:
#     classNames = f.read().rstrip("\n").split("\n")

# Load the TensorFlow Lite model
# model_path = 'model.tflite'
# interpreter = tf.lite.Interpreter(model_path=model_path)
# interpreter.allocate_tensors()
# 
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# 
# def preprocess_image(frame):
#     resized_frame = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
#     input_data = np.expand_dims(resized_frame, axis=0)
#     input_data = (input_data.astype(np.float32) - 127.5) / 127.5
#     return input_data
# 
# def getObjects(img, thres, nms, objects=[]):
#     input_data = preprocess_image(img)
# 
#     # Set the input tensor
#     interpreter.set_tensor(input_details[0]['index'], input_data)
# 
#     # Run inference
#     interpreter.invoke()
# 
#     # Get the output tensors
#     classes = interpreter.get_tensor(output_details[1]['index'])
#     scores = interpreter.get_tensor(output_details[2]['index'])
#     num_detections = int(interpreter.get_tensor(output_details[3]['index']))
# 
#     objectInfo = []
#     if num_detections != 0:
#         for i in range(num_detections):
#             if scores[0, i] > thres:
#                 className = classNames[int(classes[0, i])]
#                 if className in objects:
#                     objectInfo.append(className)


# start of keras template
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("/home/pi/Desktop/Object_Detection_Files/keras_model.h5", compile=False)

# Load the labels
class_names = open("/home/pi/Desktop/Object_Detection_Files/labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(1)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    
    if keyboard_input == 27:
        break
    

camera.release()
cv2.destroyAllWindows()



# if __name__ == "__main__":
#     cap = cv2.VideoCapture(0)
#     cap.set(3, 640)
#     cap.set(4, 480)

    #while True:
       # success, img = cap.read()

       # objectInfo = getObjects(img, 0.55, 0.2, objects=['cup', 'spoon', 'plate'])
       # print(objectInfo)

       # cv2.imshow('Object Detection', img)

        # Exit if 'q' is pressed
       # if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    # Release resources
  #  cap.release()
  #  cv2.destroyAllWindows() # ends 
#     
# def getObjects(img, thres, nms, draw=True, objects=[]):
#     classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
#     objectInfo = []
#     if len(classIds) != 0:
#         for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
#             className = classNames[classId - 1]
#             if className in objects: 
#                 objectInfo.append([box, className])
#                 if draw:
#                     cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
#                     cv2.putText(img, classNames[classId-1].upper(), (box[0] + 10, box[1] + 30),
#                                 cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#                     cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
#                                 cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#     
#     return img, objectInfo
    
#Below determines the size of the live feed window that will be displayed on the Raspberry Pi OS
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    # Variables to keep track of detected objects
    detected_spoon = False
    detected_plate = False
    detected_cup = False

    # Below is the never-ending loop that determines what will happen when an object is identified.
    # Define a dictionary to store the timestamps of detected objects
    object_timestamps = {}

    # Below is the never-ending loop that determines what will happen when an object is identified.
    while True:
        success, img = cap.read()
        # Below provides a huge amount of control. The 0.45 number is the threshold number, the 0.2 number is the nms number)
        result, objectInfo = getObjects(img, 0.45, 0.2, objects=["spoon", "plate", "cup", "hands"])
        # 
        def allOff():
            GPIO.output(solenoidCold1, 0)
            GPIO.output(solenoidCold2, 0)
            GPIO.output(solenoidHot1, 0)
            GPIO.output(solenoidHot2, 0)
        current_time = time.time() # Get current time in seconds
        
        for obj in objectInfo:
            box, className = obj
            
#             if className == "spoon" and not detected_spoon:
#                 if className not in object_timestamps or (current_time - object_timestamps[className]) > 10:
#                     # Perform command for spoon detection
#                     print("Spoon detected")
#                     GPIO.output(solenoidHot2,1)
#                     time.sleep(1)
#                     GPIO.output(solenoidHot2,0)
#                     # Add command for spoon detection here
#                     detected_spoon = True
#                     object_timestamps[className] = current_time # Update the timestamp for the detected object
#             elif className == "plate" and not detected_plate:
#                 if className not in object_timestamps or (current_time - object_timestamps[className]) > 10:
#                     # Perform command for plate detection
#                     print("Plate detected")
#                     GPIO.output(solenoidHot2,1)
#                     time.sleep(1)
#                     GPIO.output(solenoidHot2,0)
#                     # Add command for plate detection here
#                     detected_plate = True
#                     object_timestamps[className] = current_time # Update the timestamp for the detected object
#             elif className == "cup" and not detected_cup:
#                 if className not in object_timestamps or (current_time - object_timestamps[className]) > 10:
#                     # Perform command for cup detection
#                     print("Cup detected")
#                     GPIO.output(solenoidHot2,1)
#                     time.sleep(1)
#                     GPIO.output(solenoidHot2,0)
#                     # Add command for cup detection here
#                     detected_cup = True
#                     object_timestamps[className] = current_time # Update the timestamp for the detected object

            if motionWaterOff == 1:
                print("Motion Water Off")
                tempMotion = 0
               # NO OBJECT DETECTION
               
            if motionCold == 1:
                print("Motion Cold")
                tempMotion = 1
            if motionHot == 1:
                print("Motion Hot")
                tempMotion = 2
            if motionFreeFlow == 1:
                print("Motion Free Flow")
                tempMotion = 3
            while tempMotion == 1:
                allOff
            while tempMotion == 2:
                if className == "spoon" and not detected_spoon:
                    if className not in object_timestamps or (current_time - object_timestamps[className]) >10:
                        print("Spoon Detected Cold")
                        GPIO.output(solenoidCold1, 1)
                        GPIO.output(solenoidCold2, 0)
                        GPIO.output(solenoidHot1, 0)
                        GPIO.output(solenoidHot2, 0)
                        time.sleep(3)
                        allOff()
                        detected_spoon = True
                        object_timestamps[className]
                elif className == "cup" and not detected_cup:
                    if className not in object_timestamps or (current_time - object_timestamps[className]) >10:
                        print("Cup Detected Cold")
                        GPIO.output(solenoidCold1, 1)
                        GPIO.output(solenoidCold2, 1)
                        GPIO.output(solenoidHot1, 0)
                        GPIO.output(solenoidHot2, 0)
                        time.sleep(6)
                        allOff()
                        detected_cup = True
                        object_timestamps[className]
                elif className == "plate" and not detected_plate:
                    if className not in object_timestamps or (current_time - object_timestamps[className]) >10:
                        print("Plate Detected Cold")
                        GPIO.output(solenoidCold1, 1)
                        GPIO.output(solenoidCold2, 1)
                        GPIO.output(solenoidHot1, 1)
                        GPIO.output(solenoidHot2, 0)
                        time.sleep(3)
                        allOff()
                        detected_plate = True
                        object_timestamps[className]
                elif className == "hands" and not detected_hands:
                    if className not in object_timestamps or (current_time - object_timestamps[className]) >10:
                        print("Hands Detected Cold")
                        GPIO.output(solenoidCold1, 1)
                        GPIO.output(solenoidCold2, 1)
                        GPIO.output(solenoidHot1, 1)
                        GPIO.output(solenoidHot2, 0)
                        time.sleep(3)
                        allOff()
                        detected_hands = True
                        object_timestamps[className]

            while tempMotion == 3:
                if className == "spoon" and not detected_spoon:
                    if className not in object_timestamps or (current_time - object_timestamps[className]) >10:
                        print("Spoon Detected Hot")
                        GPIO.output(solenoidCold1, 0)
                        GPIO.output(solenoidCold2, 0)
                        GPIO.output(solenoidHot1, 1)
                        GPIO.output(solenoidHot2, 0)
                        time.sleep(3)
                        allOff()
                        detected_spoon = True
                        object_timestamps[className]
                elif className == "cup" and not detected_cup:
                    if className not in object_timestamps or (current_time - object_timestamps[className]) >10:
                        print("Cup Detected Hot")
                        GPIO.output(solenoidCold1, 0)
                        GPIO.output(solenoidCold2, 0)
                        GPIO.output(solenoidHot1, 1)
                        GPIO.output(solenoidHot2, 1)
                        time.sleep(3)
                        allOff()
                        detected_cup = True
                        object_timestamps[className]
                elif className == "plate" and not detected_plate:
                    if className not in object_timestamps or (current_time - object_timestamps[className]) >10:
                        print("Plate Detected Hot")
                        GPIO.output(solenoidCold1, 0)
                        GPIO.output(solenoidCold2, 1)
                        GPIO.output(solenoidHot1, 1)
                        GPIO.output(solenoidHot2, 1)
                        time.sleep(3)
                        allOff()
                        detected_plate = True
                        object_timestamps[className]
                elif className == "hands" and not detected_hands:
                    if className not in object_timestamps or (current_time - object_timestamps[className]) >10:
                        print("Hands Detected Hot")
                        GPIO.output(solenoidCold1, 0)
                        GPIO.output(solenoidCold2, 1)
                        GPIO.output(solenoidHot1, 1)
                        GPIO.output(solenoidHot2, 1)
                        time.sleep(3)
                        allOff()
                        detected_hands = True
                        object_timestamps[className]
            while tempMotion == 4:
                print("Running Water")
                GPIO.output(solenoidCold1, 1)
                GPIO.output(solenoidCold2, 1)
                GPIO.output(solenoidHot1, 1)
                GPIO.output(solenoidHot2, 1)

        # Pause for 5 seconds after detecting objects
        if detected_spoon or detected_plate or detected_cup:
            cv2.waitKey(5000)
            detected_spoon = False
            detected_plate = False
            detected_cup = False

        cv2.imshow("Output", img)
        cv2.waitKey(1)

