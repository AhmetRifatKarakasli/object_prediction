import cv2
import numpy as np
from collections import deque

dq=deque(maxlen=3)

buffer_size=16
pts=deque(maxlen=buffer_size)

bluelower=(84,98,0)
blueupper=(179,255,255)

cap=cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,480)

while True:
    success,imgoriginal=cap.read()

    if success:
        blurred=cv2.GaussianBlur(imgoriginal,(11,11),0)
        hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV Pencere",hsv)

        mask=cv2.inRange(hsv,bluelower,blueupper)
        mask=cv2.erode(mask,None,iterations=2)
        mask=cv2.dilate(mask,None,iterations=2)
        cv2.imshow("Mask Pencere",mask)

        (contours,_)=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        center=None

        if len(contours)>0:
            c=max(contours,key=cv2.contourArea)
            rect=cv2.minAreaRect(c)
            ((x,y),(width,height),rotation)=rect
            s="x:{}, y:{}, width:{}, height:{}, rotation:{}".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
            print(s)

            box=cv2.boxPoints(rect)
            box=np.int64(box)

            M=cv2.moments(c)
            center=(int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
            cv2.drawContours(imgoriginal,[box],0,(0,255,255),2)
            cv2.circle(imgoriginal,center,5,(255,0,0))

        pts.append(center)
        for i in range(1,len(pts)):
            if pts[i-1] is None or pts[i] is None:continue
            cv2.line(imgoriginal,pts[i-1],pts[i],(0,255,255),3)
        cv2.imshow("Orjinal",imgoriginal)

    if cv2.waitKey(10) & 0xFF==ord("q"):
        break












