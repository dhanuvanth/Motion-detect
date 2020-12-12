# imports
import numpy as np
import cv2
import os,sys,time


# init camera
camera = cv2.VideoCapture(0)   # Live Stream
camera.set(3,1280)             # frame width
camera.set(4,720)              # frame height
time.sleep(0.5)

copyFrame = None
while True:

    # grab a frame
    ret,frame = camera.read()
    
    # end of feed
    if not ret:
        break

    # gray frame
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # blur frame
    blur = cv2.GaussianBlur(gray,(15,15),0)

    # initialize copyFrame
    if copyFrame is None:
        copyFrame = blur
        continue

    frame2 = cv2.absdiff(copyFrame,blur)

    # threshold frame
    thrs = cv2.threshold(frame2,15,255,cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes
    kernel = np.ones((2,2),np.uint8)
    kp = cv2.erode(thrs,kernel,iterations=4)
    kp = cv2.dilate(kp,kernel,iterations=8)

    # find contours on thresholded image
    _,contours,_ = cv2.findContours(kp.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # make coutour frame
    frame_contour = frame.copy()

    # target contours
    targets = []

    # loop over the contours
    for c in contours:
        
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 400:
                continue

        # contour data
        M = cv2.moments(c)#;print( M )
        print(M)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        x,y,w,h = cv2.boundingRect(c)
        ca = cv2.contourArea(c)

        # plot contours
        cv2.drawContours(frame_contour,[c],0,(0,0,255),2)

        # save target contours
        targets.append((cx,cy,ca))

    # make target
    mx = 0
    my = 0
    if targets:

        # centroid of largest contour
        area = 0
        for x,y,a in targets:
            if a > area:
                mx = x
                my = y
                area = a
        
    
    # update master
    copyFrame = blur

    # display
    cv2.imshow("Frame0: Raw",frame)
    cv2.imshow("Frame6: Contours",frame_contour)
    
    # key delay and action
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# release camera
camera.release()

# close all windows
cv2.destroyAllWindows()