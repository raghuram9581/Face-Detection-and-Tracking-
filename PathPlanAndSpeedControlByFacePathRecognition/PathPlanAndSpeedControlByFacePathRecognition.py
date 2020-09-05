### Author @ Raghuram Vadlamani ###

import cv2
import numpy as np
import time
#import RPi.GPIO as GPIO
try:
    
    #from pyfirmata import Arduino, util
    import pyfirmata as fir
except:
    import pip
    pip.main(['install','pyfirmata'])
    from pyfirmata import Arduino, util
import time


mid_Objpre1 = (0,0)
mid_Obj = (0,0)


def scaleBetween(unscaledNum, minAllowed, maxAllowed, min, max):

    return (maxAllowed - minAllowed) * (unscaledNum - min) / (max - min) + minAllowed


#GPIO.setmode(GPIO.BOARD)
#GPIO.setup(11,GPIO.OUT)
#servo1 = GPIO.PWM(11,50)
#GPIO.setup(12,GPIO.OUT)
#servo1 = GPIO.PWM(12,50)

a = fir.Arduino('COM4')
iterator = fir.util.Iterator(a)
iterator.start()

s_m1 = a.get_pin('d:10:p')
d_m1 = a.get_pin('d:8:o')
s_m2 = a.get_pin('d:9:p')
d_m2 = a.get_pin('d:7:o')

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

_, frame = cap.read()

frame_x = frame.shape[0]
frame_y = frame.shape[1]
print(frame_x)
print(frame_y)

mid_x = frame_x/2
print(mid_x)
mid_y = frame_y/2
print(mid_y)

mid_frame = (int(mid_x),int(mid_y))
print(mid_frame)

while True:
    _, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(frame_gray,scaleFactor=1.05,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    for x,y,h,w in faces:
        mid_Obj_x = (x+x+w)/2
        mid_Obj_y = (y+y+h)/2
        mid_Objpre1 = mid_Obj
        mid_Obj = (int(mid_Obj_x),int(mid_Obj_y))
        mid_Obj = np.array([int(mid_Obj_x),int(mid_Obj_y)])
        print("current")
        #print(mid_Obj)
        
        
        cv2.circle(frame,(int(mid_Obj_x),int(mid_Obj_y)),3,(0,0,255),2)
        cv2.putText(frame, str(mid_Obj), (int(mid_Obj_x),int(mid_Obj_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2, cv2.LINE_AA)
        cv2.circle(frame,(int(mid_y),int(mid_x)),3,(0,0,255),2)
        cv2.circle(frame,(320,240),3,(0,0,255),2)
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        

        
####################################################################################        
        #scale_x = scaleBetween(int(mid_Obj_x),0,180,0,640)
        #print("Angle of servo in X direction")
        #print(scale_x)
        #servo1.ChangeDutyCycle(2+(scale_x/18))
        #scale_y = scaleBetween(int(mid_Obj_y),0,180,0,480)
        #print("Angle of servo in Y direction")
        #print(scale_y)
        #servo1.ChangeDutyCycle(2+(scale_y/18))
####################################################################################
        if (mid_x < mid_Obj_x):
            s_1 = scaleBetween(int(mid_Obj_x),0,1,320,640)
            #cv2.putText(frame, int(s), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2, cv2.LINE_AA)
            print("s_1")
            print(s_1)
            print(abs(s_1))
            s_m1.write(abs(s_1))
            d_m1.write(0)
            s_m2.write(abs(s_1))
            d_m2.write(0)

            
        elif (mid_x >= mid_Obj_x):
            s_2 = scaleBetween(int(mid_Obj_x),0,1,320,0)
            #cv2.putText(frame, int(s), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2, cv2.LINE_AA)
            print("s_2")
            print(s_2)
            print(abs(s_2))
            s_m1.write(abs(s_2))
            d_m1.write(1)
            s_m2.write(abs(s_2))
            d_m2.write(1)
        

   # print(frame.shape)
        
   
        pts = np.array(mid_Objpre1)
        pts=np.vstack((pts,mid_Obj))
        #print("previous")
        #print(pts)
        cv2.polylines(frame, np.int32([pts]), False, (0,255,0), 5)
    cv2.imshow("frame",frame)

   
    

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()



