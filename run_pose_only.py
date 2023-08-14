import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import time
import numpy as np
from src2 import utils_pose_only as utils
# import argparse
        
def pose_detector_callback(result, output_frame, timestamp):
    global annotated_frame
    #t = time.time()
    annotated_frame = np.copy(cv2.cvtColor(output_frame.numpy_view(), cv2.COLOR_RGB2BGR))
    height, width = annotated_frame.shape[:2]
    landmark_names = ('l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist')
    
    landmark_dict = dict()
    pose_landmarks_list = result.pose_landmarks
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        for i , landmark_name in enumerate(landmark_names, 11):
            landmark_cordinate = np.array([int(pose_landmarks[i].x * width), int(pose_landmarks[i].y * height)])
            #print(pose_landmarks)
            landmark_dict[landmark_name] = landmark_cordinate
           
            landmark_dict['l_visibility'] = pose_landmarks[15].visibility
            landmark_dict['r_visibility'] = pose_landmarks[16].visibility
        
        # print(landmark_dict)
        #print('l',timestamp,l_gesture_dict)
        #print('r',timestamp,r_gesture_dict)
        #utils.arm_operation(landmark_dict, annotated_frame, obj_dict, l_gesture_dict, r_gesture_dict, timestamp)
        utils.arm_operation(landmark_dict, annotated_frame, obj_dict)
            
       
            
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x,y=landmark.y, z=landmark.z) for landmark in pose_landmarks])
        solutions.drawing_utils.draw_landmarks(annotated_frame,
                                            pose_landmarks_proto,
                                            solutions.pose.POSE_CONNECTIONS,
                                            solutions.drawing_styles.get_default_pose_landmarks_style())
  
   
  
def main():
    global annotated_frame
    scale = 2
    WIDTH = 640*scale
    HEIGHT = 360*scale
    
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        
    if not cap.isOpened():
        print("Cannot open a video capture.")
        exit(-1)
        
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


    print(f'resolution:{width}x{height}')
    print('FPS:' ,cap.get(cv2.CAP_PROP_FPS))
    
    POSE_DETECTOR_MODEL = 'pose_landmarker_lite.task'
    POSE_DETECTOR_MODEL = 'pose_landmarker_full.task'
    # POSE_DETECTOR_MODEL = 'pose_landmarker_heavy.task'
    
    
    pose_detector_options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=POSE_DETECTOR_MODEL),
                                                         running_mode=vision.RunningMode.LIVE_STREAM,
                                                         result_callback=pose_detector_callback)

    pose_detector =  vision.PoseLandmarker.create_from_options(pose_detector_options)
    # gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_recognizer_options)
    global obj_dict
    obj_dict = {'tv':(0, 200*scale, 100*scale, 160*scale), 'fan':(550*scale, 260*scale, 90*scale, 100*scale)}
    # ret, annotated_frame_list[0] = cap.read()
    timestamp = 0
    pre_t = time.time()
    #annotated_frame_list = list()
    ret, frame = cap.read()
    annotated_frame = np.copy(frame)
    #show_timestamp = 0
    while True:
        t = time.time() - pre_t
        pre_t = time.time()
       
        
        try:
            fps = 1/t
        except ZeroDivisionError:
            fps = 0
    
        ret, frame = cap.read()
        
        if not ret:
            print('error')
            break
        #frame = cv2.flip(frame, 1)
        
        
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        pose_detector.detect_async(mp_frame, timestamp)
        
        # gesture_recognizer.recognize_async(mp_frame, timestamp)
        
        #print('timestamp',timestamp)
        
        #print('FPS', fps)
        #cv2.putText(annotated_frame, f'FPS:{fps:.1f}', (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        for name, (x, y, w, h) in obj_dict.items():
            #print(x,y,w,h)
            cv2.putText(annotated_frame, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(annotated_frame, (x,y),(x+w, y+h), (0,0,255), 2)

       
        cv2.imshow('frame', annotated_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        timestamp += 1
        #annotated_frame = np.copy(frame)
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
