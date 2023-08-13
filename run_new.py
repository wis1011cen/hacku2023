import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import time
import numpy as np
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# import argparse

def gesture_recognizer_callback(result, output_frame, timestamp):
    global annotated_frame
    #gesture_timestamp = timestamp
    #t = time.time()
    # annotated_frame = np.copy(cv2.cvtColor(output_frame.numpy_view(), cv2.COLOR_RGB2BGR))
    gestures =  result.gestures
    handedness = result.handedness
    # hand_landmarks = 
    multi_hand_landmarks_list = [multi_hand_landmarks for multi_hand_landmarks in result.hand_landmarks]
    
    gesture_dict = dict()
    if gestures:
        for i, (hand, gesture) in enumerate(zip(handedness, gestures)):
            gesture_dict[hand[0].category_name] = gesture[0].category_name
            cv2.putText(annotated_frame, f'{hand[0].category_name}:{gesture[0].category_name}', (0, 60+30*i), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            
'''
    for hand_landmarks in multi_hand_landmarks_list:
        # print(hand_landmarks)
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])

        solutions.drawing_utils.draw_landmarks(annotated_frame_list[timestamp],
                                  hand_landmarks_proto,
                                  solutions.hands.HAND_CONNECTIONS,
                                  solutions.drawing_styles.get_default_hand_landmarks_style(),
                                  solutions.drawing_styles.get_default_hand_connections_style())
        
        # cv2.imshow('video', annotated_frame)
        # cv2.waitKey(1)
    print('gesture', timestamp, time.time()-t)
    # annotated_frame = np.copy(cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_RGB2BGR))
'''
    

        
def pose_detector_callback(result, output_frame, timestamp):
    global annotated_frame
    #pose_timestamp = timestamp
    #t = time.time()
    # annotated_frame = np.copy(cv2.cvtColor(output_frame.numpy_view(), cv2.COLOR_RGB2BGR))
    height, width = annotated_frame.shape[:2]
    pose_landmarks_list = result.pose_landmarks
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        for i in range(len(pose_landmarks)):
            lm_pos = np.array([int(pose_landmarks[i].x * width), int(pose_landmarks[i].y * height)])
            # print(i, lm_pos)
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x,y=landmark.y, z=landmark.z) for landmark in pose_landmarks])
        solutions.drawing_utils.draw_landmarks(annotated_frame,
                                               pose_landmarks_proto,
                                               solutions.pose.POSE_CONNECTIONS,
                                               solutions.drawing_styles.get_default_pose_landmarks_style())
   
    #print('pose', timestamp,time.time()-t)
    
#annotated_frame = None
#gesture_timestamp = 0
#pose_timestamp=0
def main():
    global annotated_frame
    WIDTH = 640
    HEIGHT = 360
    scale = 1
    
    cap = cv2.VideoCapture(0)

    if not (cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH) and cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        scale = 2
   
    
        
    if not cap.isOpened():
        print("Cannot open a video capture.")
        exit(-1)
        
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


    print(f'resolution:{width}x{height}')
    
    
    pose_detector_options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='pose_landmarker_lite.task'),
                                                         running_mode=vision.RunningMode.LIVE_STREAM,
                                                         result_callback=pose_detector_callback)
    gesture_recognizer_options = vision.GestureRecognizerOptions(num_hands=2,
                                                                 base_options=python.BaseOptions(model_asset_path='gesture_recognizer.task'),
                                                                 running_mode=vision.RunningMode.LIVE_STREAM,
                                                                 result_callback=gesture_recognizer_callback)

    pose_detector =  vision.PoseLandmarker.create_from_options(pose_detector_options)
    gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_recognizer_options)
    
    # ret, annotated_frame_list[0] = cap.read()
    timestamp = 0
    roi_dict = {'tv':(0, 200*scale, 100*scale, 160*scale), 'fan':(550*scale, 260*scale, 90*scale, 100*scale)}
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
        frame = cv2.flip(frame, 1)
        
        
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        pose_detector.detect_async(mp_frame, timestamp)
        # annotated_frame
        gesture_recognizer.recognize_async(mp_frame, timestamp)
        
        #print('timestamp',timestamp)
        
        
        cv2.putText(annotated_frame, f'FPS:{fps:.1f}', (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        # for name, (x,y,w,h) in roi_dict.items():
        #     cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        #     cv2.rectangle(frame, (x,y),(x+w, y+h), (0,0,255), 2)
        #print('timestamp','list size',timestamp,len(annotated_frame_list))
        #if gesture_timestamp >= show_timestamp and pose_timestamp >= show_timestamp:
            #print('show',show_timestamp,gesture_timestamp, pose_timestamp)
        cv2.imshow('frame', annotated_frame)
        
            #show_timestamp += 1
            
        
        timestamp += 1
        
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        annotated_frame = np.copy(frame)
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
