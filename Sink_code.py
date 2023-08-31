solenoidCold1 = 5
solenoidCold2 = 6
solenoidHot1 = 22
motionFreeFlow = 16
motionWarm = 19
motionCold = 20
motionWaterOff = 26
tempMotion = 0

GPIO.setmode(GPIO.BCM)
GPIO.setup(solenoidHot1, GPIO.OUT)
GPIO.output(solenoidHot1, 0)
GPIO.setup(solenoidCold1, GPIO.OUT)
GPIO.output(solenoidCold1, 0)
GPIO.setup(solenoidCold2, GPIO.OUT)
GPIO.output(solenoidCold2, 0)
GPIO.setup(motionFreeFlow, GPIO.IN)
GPIO.input(motionFreeFlow, 0)
GPIO.setup(motionWarm, GPIO.IN)
GPIO.input(motionWarm, 0)
GPIO.setup(motionCold, GPIO.IN)
GPIO.input(motionCold, 0)
GPIO.setup(motionWaterOff, GPIO.IN)
GPIO.input(motionWaterOff, 0)

if motionWaterOff == 1
	tempMotion = 1
	NO OBJECT DETECTION
if motionCold == 1
	tempMotion = 2
if motionHot == 1
	tempMotion = 3
if motionFreeFlow == 1
	tempMotion = 4
while tempMotion == 1
	GPIO.output(solenoidHot1, 0)
	GPIO.output(solenoidHot2, 0)
	GPIO.output(solenoidCold1, 0)
	GPIO.output(solenoidCold2, 0)
while tempMotion == 2
	if className == "spoon" and not detected_spoon:
		if className not in object_timestamps or (current_time - object_timestamps[className]) >10;
			print("Spoon Detected Cold")
				GPIO.output(solenoidCold1, 1)
				GPIO.output(solenoidCold2, 0)
				GPIO.output(solenoidHot1, 0)
				GPIO.output(solenoidHot2, 0)
				time.sleep(3)
				GPIO.output(solenoidCold1, 0)
				GPIO.output(solenoidCold2, 0)
				GPIO.output(solenoidHot1, 0)
				GPIO.output(solenoidHot2, 0)
				detected_spoon = True
				object_timestamps[className]
	elif className == "cup" and not detected_cup:
		if className not in object_timestamps or (current_time - object_timestamps[className]) >10;
			print("Cup Detected Cold")
				GPIO.output(solenoidCold1, 1)
				GPIO.output(solenoidCold2, 1)
				GPIO.output(solenoidHot1, 0)
				GPIO.output(solenoidHot2, 0)
				time.sleep(3)
				GPIO.output(solenoidCold1, 0)
				GPIO.output(solenoidCold2, 0)
				GPIO.output(solenoidHot1, 0)
				GPIO.output(solenoidHot2, 0)
				detected_cup = True
				object_timestamps[className]
	elif className == "plate" and not detected_plate:
		if className not in object_timestamps or (current_time - object_timestamps[className]) >10;
			print("Plate Detected Cold")
				GPIO.output(solenoidCold1, 1)
				GPIO.output(solenoidCold2, 1)
				GPIO.output(solenoidHot1, 1)
				GPIO.output(solenoidHot2, 0)
				time.sleep(3)
				GPIO.output(solenoidCold1, 0)
				GPIO.output(solenoidCold2, 0)
				GPIO.output(solenoidHot1, 0)
				GPIO.output(solenoidHot2, 0)
				detected_plate = True
				object_timestamps[className]
	elif className == "hands" and not detected_hands:
		if className not in object_timestamps or (current_time - object_timestamps[className]) >10;
			print("Hands Detected Cold")
				GPIO.output(solenoidCold1, 1)
				GPIO.output(solenoidCold2, 1)
				GPIO.output(solenoidHot1, 1)
				GPIO.output(solenoidHot2, 0)
				time.sleep(3)
				GPIO.output(solenoidCold1, 0)
				GPIO.output(solenoidCold2, 0)
				GPIO.output(solenoidHot1, 0)
				GPIO.output(solenoidHot2, 0)
				detected_hands = True
				object_timestamps[className]

while tempMotion == 3
	if className == "spoon" and not detected_spoon:
		if className not in object_timestamps or (current_time - object_timestamps[className]) >10;
			print("Spoon Detected Hot")
				GPIO.output(solenoidCold1, 0)
				GPIO.output(solenoidCold2, 0)
				GPIO.output(solenoidHot1, 1)
				GPIO.output(solenoidHot2, 0)
				time.sleep(3)
				GPIO.output(solenoidCold1, 0)
				GPIO.output(solenoidCold2, 0)
				GPIO.output(solenoidHot1, 0)
				GPIO.output(solenoidHot2, 0)
				detected_spoon = True
				object_timestamps[className]
	elif className == "cup" and not detected_cup:
		if className not in object_timestamps or (current_time - object_timestamps[className]) >10;
			print("Cup Detected Hot")
				GPIO.output(solenoidCold1, 0)
				GPIO.output(solenoidCold2, 0)
				GPIO.output(solenoidHot1, 1)
				GPIO.output(solenoidHot2, 1)
				time.sleep(3)
				GPIO.output(solenoidCold1, 0)
				GPIO.output(solenoidCold2, 0)
				GPIO.output(solenoidHot1, 0)
				GPIO.output(solenoidHot2, 0)
				detected_cup = True
				object_timestamps[className]
	elif className == "plate" and not detected_plate:
		if className not in object_timestamps or (current_time - object_timestamps[className]) >10;
			print("Plate Detected Hot")
				GPIO.output(solenoidCold1, 0)
				GPIO.output(solenoidCold2, 1)
				GPIO.output(solenoidHot1, 1)
				GPIO.output(solenoidHot2, 1)
				time.sleep(3)
				GPIO.output(solenoidCold1, 0)
				GPIO.output(solenoidCold2, 0)
				GPIO.output(solenoidHot1, 0)
				GPIO.output(solenoidHot2, 0)
				detected_plate = True
				object_timestamps[className]
	elif className == "hands" and not detected_hands:
		if className not in object_timestamps or (current_time - object_timestamps[className]) >10;
			print("Hands Detected Hot")
				GPIO.output(solenoidCold1, 0)
				GPIO.output(solenoidCold2, 1)
				GPIO.output(solenoidHot1, 1)
				GPIO.output(solenoidHot2, 1)
				time.sleep(3)
				GPIO.output(solenoidCold1, 0)
				GPIO.output(solenoidCold2, 0)
				GPIO.output(solenoidHot1, 0)
				GPIO.output(solenoidHot2, 0)
				detected_hands = True
				object_timestamps[className]
while tempMotion == 4
	print("Running Water")
	GPIO.output(solenoidCold1, 1)
	GPIO.output(solenoidCold2, 1)
	GPIO.output(solenoidHot1, 1)
    GPIO.output(solenoidHot2, 1)

