import cv2
import numpy as np
capl = cv2.VideoCapture(0)
capr = cv2.VideoCapture(1)
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

while True:
            
    retl, framel = capl.read()
    retr, framer = capr.read()
    
    
    print(retl, retr)
    #print(framel.shape, framer.shape)
    
    framel = cv2.cvtColor(framel, cv2.COLOR_BGR2GRAY)
    framer = cv2.cvtColor(framer, cv2.COLOR_BGR2GRAY)
        
    if not retl:
        print('error')
        break
    
    
       
    disparity = stereo.compute(framel, framer)
    THRESHOLD = -1000
    disparity = np.where(disparity < THRESHOLD, THRESHOLD, disparity)
    disparity = np.where(disparity > 1000, 1000, disparity)
            
    disparity = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    disparity = np.uint8(disparity)


    cv2.imshow('video', disparity)
        
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
            
    
cv2.destroyAllWindows()
capl.release()