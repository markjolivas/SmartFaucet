import cv2
import time
import RPi.GPIO as GPIO

solenoidHot2 = 6

GPIO.setmode(GPIO.BCM)
GPIO.setup(solenoidHot2, GPIO.OUT)
GPIO.output(solenoidHot2, False)
GPIO.output(solenoidHot2, 0)


GPIO.output(solenoidHot2,1)
time.sleep(1)
GPIO.output(solenoidHot2,0)