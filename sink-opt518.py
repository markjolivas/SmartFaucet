import cv2
import time
import RPi.GPIO as GPIO
import numpy as np
from keras.models import load_model


# GPIO pins initialization
solenoidHot2 = 5
solenoidCold1 = 27
solenoidCold2 = 22
solenoidHot1 = 6
motionFreeFlow = 21
motionHot = 19
motionCold = 16
motionWaterOff = 20
tempMotion = 0

GPIO.setmode(GPIO.BCM)
GPIO.setup(
    [solenoidHot2, solenoidHot1, solenoidCold1, solenoidCold2],
    GPIO.OUT,
    initial=0
)
GPIO.setup(
    [motionFreeFlow, motionHot, motionCold, motionWaterOff],
    GPIO.IN
)
GPIO.setwarnings(False)

detected_spoon = False
detected_plate = False
detected_cup = False

detected_hands = False

# Load the model
model = load_model("/home/pi/Desktop/Object_Detection_Files/keras_model.h5", compile=False)

# Load the labels
class_names = "/home/pi/Desktop/Object_Detection_Files/labels.txt"
with open(class_names, 'r') as f:
    class_name = f.read().rstrip("\n").split("\n")

# CAMERA can be 0 or 1 based on the default camera of your computer
camera = cv2.VideoCapture(1)

# Define a dictionary to store the timestamps of detected objects
object_timestamps = {}

def allOff():
    GPIO.output(solenoidCold1, 0)
    GPIO.output(solenoidCold2, 0)
    GPIO.output(solenoidHot1, 0)
    GPIO.output(solenoidHot2, 0)
    
def allOn():
    GPIO.output(solenoidCold1, 1)
    GPIO.output(solenoidCold2, 1)
    GPIO.output(solenoidHot1, 1)
    GPIO.output(solenoidHot2, 1)

def process_frame(image):
    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predict the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]

    return index, confidence_score

empty = 0
spoon = 1
cup = 2
plate = 3
hands = 4

while True:
    # Grab the web camera's image.
    ret, image = camera.read()

    if not ret:
        break

    # Perform object detection
    index, confidence_score = process_frame(image)

    current_time = time.time()

    # TO-DO: 4 CASES OF NO WATER, WATER, HOT, AND COLD FOR THE MOTION
    # DE-ACTIVATE OBJ DETECTION WHILE MOTION ON AND OFF
    # IMPLEMENT MOTION SENSORS...
    # TIMING MECHANISM
    # SEE TESTGPT FOR FINISHED MOTION CODE... FIX NAMING
    waterOff = GPIO.input(20)
    freeFlow = GPIO.input(21)
    cold = GPIO.input(16)
    hot = GPIO.input(19)

    if waterOff == 1:
        tempMotion = 1
        print("Motion Water Off")
        allOff()
        time.sleep(3)
    elif freeFlow == 1:
        tempMotion = 1
        allOn()
        print("Motion Water All On")
        time.sleep(3)
    elif cold == 1:
        allOff()
        tempMotion = 2
        print("Temp Motion 2")
    elif hot == 1:
        allOff()
        tempMotion = 3
        print("Temp Motion 3")

    if tempMotion == 2:
        print("Motion Water Cold")
        if index == spoon and not detected_spoon:
            print("Cold spoon")
            if index not in object_timestamps or (current_time - object_timestamps[index]) > 10:
                GPIO.output(solenoidCold1, 1)
                time.sleep(3)
                allOff()
                detected_spoon = True
                object_timestamps[index] = current_time
        elif index == cup and not detected_cup:
            print("Cold cup")
            if index not in object_timestamps or (current_time - object_timestamps[index]) > 10:
                GPIO.output(solenoidCold1, 1)
                GPIO.output(solenoidCold2, 1)
                time.sleep(3)
                allOff()
                detected_cup = True
                object_timestamps[index] = current_time
        elif index == plate and not detected_plate:
            print("Cold plate")
            if index not in object_timestamps or (current_time - object_timestamps[index]) > 10:
                GPIO.output(solenoidCold1, 1)
                GPIO.output(solenoidCold2, 1)
                GPIO.output(solenoidHot1, 1)
                time.sleep(3)
                allOff()
                detected_plate = True
                object_timestamps[index] = current_time
        elif index == hands and not detected_hands:
            print("Cold hands")
            if index not in object_timestamps or (current_time - object_timestamps[index]) > 10:
                GPIO.output(solenoidCold1, 1)
                GPIO.output(solenoidCold2, 1)
                time.sleep(3)
                allOff()
                detected_hands = True
                object_timestamps[index] = current_time

    elif tempMotion == 3:
        print("Motion Water Hot")
        if index == spoon and not detected_spoon:
            print("Hot spoon")
            if index not in object_timestamps or (current_time - object_timestamps[index]) > 10:
                GPIO.output(solenoidCold1, 1)
                time.sleep(3)
                allOff()
                detected_spoon = True
                object_timestamps[index] = current_time
        elif index == cup and not detected_cup:
            print("Hot cup")
            if index not in object_timestamps or (current_time - object_timestamps[index]) > 10:
                GPIO.output(solenoidCold1, 1)
                GPIO.output(solenoidCold2, 1)
                time.sleep(3)
                allOff()
                detected_cup = True
                object_timestamps[index] = current_time
        elif index == plate and not detected_plate:
            print("Hot plate")
            if index not in object_timestamps or (current_time - object_timestamps[index]) > 10:
                GPIO.output(solenoidCold1, 1)
                GPIO.output(solenoidCold2, 1)
                GPIO.output(solenoidHot1, 1)
                time.sleep(3)
                allOff()
                detected_plate = True
                object_timestamps[index] = current_time
        elif index == hands and not detected_hands:
            print("Hot hands")
            if index not in object_timestamps or (current_time - object_timestamps[index]) > 10:
                GPIO.output(solenoidCold1, 1)
                GPIO.output(solenoidCold2, 1)
                time.sleep(3)
                allOff()
                detected_hands = True
                object_timestamps[index] = current_time

    if detected_spoon or detected_plate or detected_cup or detected_hands:
        cv2.waitKey(5000)
        detected_spoon = False
        detected_plate = False
        detected_cup = False
        detected_hands = False
        camera.read()

    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
