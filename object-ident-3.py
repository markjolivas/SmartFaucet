import cv2

import time

import RPi.GPIO as GPIO

solenoidHot2 = 22

GPIO.setmode(GPIO.BCM)

GPIO.setup(solenoidHot2, GPIO.OUT)
#GPIO.output(solenoidHot2, True)
GPIO.output(solenoidHot2, False)
GPIO.output(solenoidHot2, 1)
GPIO.output(solenoidHot2, 0)

#GPIO.setmode(GPIO.BCM)
#GPIO.setup(LED_PIN, GPIO.OUT)
#GPIO.output(LED_PIN, GPIO.LOW)

#from gpiozero import AngularServo
#servo =AngularServo(18, initial_angle=0, min_pulse_width=0.0004, max_pulse_width=0.0005)
#thres = 0.45 # Threshold to detect object

classNames = []
classFile = "/home/bd-sink/Desktop/Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/bd-sink/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/bd-sink/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects: 
                objectInfo.append([box,className])
#                 if (draw):
#                     cv2.rectangle(img,box,color=(0,255,0),thickness=2)
#                     cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
#                     cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
#                     cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
#                     cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    
                    #servo.angle = -90
                    #time.sleep = 2
                    #servo.angle = 90
    
    return img,objectInfo


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(2,240)
    cap.set(2,240)
    #cap.set(10,70)
    
    running = True
    
    
    while running:
        success, img = cap.read()
        
        
        result, objectInfo = getObjects(img,0.55,0.2, objects=['cup', 'spoon', 'plate'])
        #return objects
        obj911 = [sub[1] for sub in objectInfo if 'cup' in sub[1]]
    #while(objects =='cup'):    
        #print(str(obj911))
        print(obj911)
        #print(1)
              
        
        cv2.imshow("Output",img)
        cv2.waitKey(1)
        
        key = cv2.waitKey(1)
        
        #if key & 0xFF == ord('q'):
            
        if (obj911 ==['cup']):
            
           
            
            #for (i = 0; i >200; i++)#while True:
            GPIO.output(solenoidHot2,1)
           # getObjects (img,0.55,0.2, objects=['moose'])
            #cv2.imshow()
            #cv2.waitKey(0)
            
            
            time.sleep(2)
            
            key = cv2.waitKey(1)
            if key == ord('p'):
                while True:
                    key = cv2.waitKey(1)
                    if key == ord('r'):
                        break
            
            
            time.sleep(1)
            print ("stop")
            running = False
            time.sleep(5)
            running = True
            
            #cv2.waitKey(1)
            
            #obj911 = [0]
           # img = cap.read()
            #GPIO.output(solenoidHot2,0)
            #time.sleep = 20

           # while(obj911 == ['cup']): solenoidHot2 = 1; time.sleep(2); 
            
        else:
            GPIO.output(solenoidHot2,0)            
            


            #GPIO.cleanup()
                    
            
#             cv2.imshow("Output",img)
#             cv2.waitKey(1)
#

#[[array([216, 197, 239, 261], dtype=int32), 'cup']]

# 
# ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']
# ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']
# ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']

