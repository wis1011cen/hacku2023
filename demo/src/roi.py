import cv2
import csv
import src.utils as utils

def select_roi(cap, winname, flip_flag):
    roi_dict= {}
    
    while True:
        ret, frame = cap.read()
        if flip_flag:
            frame = cv2.flip(frame, 1)
        canvas = frame.copy()
        cv2.putText(canvas, "Press 'r' to select a ROI.", (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(winname, canvas)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            canvas1 = frame.copy()
            canvas2 = frame.copy()
            while True:
                name = input_name(winname, canvas1)
                
                if name is None:    # ESCが押されたとき
                    cv2.destroyAllWindows()
                
                        
                    return roi_dict
                cv2.putText(canvas2, 'Select a ROI and then press ENTER.', (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                roi = cv2.selectROI(winname, canvas2, showCrosshair=False, fromCenter=False)
                x, y, w, h = map(int, roi)
                
                roi_dict[name] = (x, y, w, h)
                
                #　名前と矩形表示
                cv2.putText(canvas1, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(canvas2, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                
                cv2.rectangle(canvas1, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.rectangle(canvas2, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        
        elif key == ord('q'):
            cv2.destroyAllWindows()
            break
        
def input_name(winname, canvas):
    canvas_copy = canvas.copy()
    name = str()
    while True:
        cv2.putText(canvas_copy, 'input name:', (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow(winname, canvas_copy)
        key = cv2.waitKey(0)
        if key == 13:                               # Enter
            return name
        elif (key >= 48 and key <= 57) or (key >= 97 and key <= 122):              # 0-9, a-z
            char = chr(key)
            name += char
        elif key == 127 and len(name) != 0:         # backspace
            canvas_copy = canvas.copy()
            name = name [:-1]
        elif key == 27:                             # ESC
            return None
        cv2.putText(canvas_copy, name, (230, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            
                

def roi(flip_flag):
    cap = cv2.VideoCapture(0)
    SCALE = utils.SCALE
    WIDTH = 640*SCALE
    HEIGHT = 360*SCALE
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    roi_dict = select_roi(cap, 'video', flip_flag)
    print(roi_dict)
    #appliance_dict = roi.select_roi(cap, 'video', args.flip)
    roi_list = []
    try:
        for name, (x, y, w, h) in roi_dict.items():
            roi_list.append([name, x, y, w, h])
            
        with open('src/roi.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(roi_list)
    except:
        pass
    """
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        for name, (x,y,w,h) in roi_dict.items():
            cv2.putText(frame, name, (x, y-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
            cv2.rectangle(frame, (x,y),(x+w, y+h),(0,0,255))

        cv2.imshow('video', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
        """
