import cv2
import time
import RPi.GPIO as GPIO
import numpy as np                      
import tensorflow as tf
from keras.models import load_model

# GPIO pins initialization
solenoidHot2 = 27
solenoidCold1 = 5
solenoidCold2 = 6
solenoidHot1 = 22
motionFreeFlow = 16
motionHot = 19
motionCold = 21
motionWaterOff = 26

GPIO.setmode(GPIO.BCM)
GPIO.setup([solenoidHot2, solenoidHot1, solenoidCold1, solenoidCold2], GPIO.OUT, initial=0)
GPIO.setup([motionFreeFlow, motionHot, motionCold, motionWaterOff], GPIO.IN)

# Load the model and labels
model = load_model("/home/pi/Desktop/Object_Detection_Files/keras_model.h5", compile=False)
class_names = open("/home/pi/Desktop/Object_Detection_Files/labels.txt", "r").readlines()
class_index_to_name = {i: name.strip() for i, name in enumerate(class_names)}

# Open camera
camera = cv2.VideoCapture(1)

while True:
    # Capture image
    ret, image = camera.read()

    # Preprocess image
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    # Predict class
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    #print(f"Class: {class_name}, Confidence Score: {confidence_score:.2%}")
    
    def allOff():
        GPIO.output(solenoidCold1, 0)
        GPIO.output(solenoidCold2, 0)
        GPIO.output(solenoidHot1, 0)
        GPIO.output(solenoidHot2, 0)
    
    # Variables to keep track of detected objects
    detected_spoon = False
    detected_plate = False
    detected_cup = False

    # Below is the never-ending loop that determines what will happen when an object is identified.
    # Define a dictionary to store the timestamps of detected objects
#     object_timestamps = {}
#     for obj in objectInfo:
#             box, className = obj    
#             if class_name == "spoon" and not detected_spoon:
#                 if class_name not in object_timestamps or (current_time - object_timestamps[class_name]) >10:
#                     print("Spoon Detected Cold")
#                     GPIO.output(solenoidCold1, 1)
#                     GPIO.output(solenoidCold2, 0)
#                     GPIO.output(solenoidHot1, 0)
#                     GPIO.output(solenoidHot2, 0)
#                         time.sleep(3)
#                     allOff()
#                         detected_spoon = True
  
    # Check for keyboard input
    # Pause for 5 seconds after detecting objects
if detected_spoon or detected_plate or detected_cup:
    cv2.waitKey(5000)
    detected_spoon = False
    detected_plate = False
    detected_cup = False
    # Listen to the keyboard for presses.

camera.release()
cv2.destroyAllWindows()


