# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 19:40:10 2019

@author: ASUS
"""

import cv2, time
video=cv2.VideoCapture(0)

while True:    
    check, frame = video.read()
    
    print(check)
    print(frame)
    
    cv2.imshow("Video streaming", frame)
    key=cv2.waitKey(1) & 0b11111111
    
    if key == ord('q'):
        break
    

video.release()
cv2.destroyWindow("Video streaming")