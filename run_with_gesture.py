import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import time
import numpy as np
import src.utils as utils

CATEGORY_ALLOWLIST = ['None', 'Thumb_Up', 'Thumb_Down']

def gesture_recognizer_callback(result, output_frame, timestamp):
    global l_gesture_dict, r_gesture_dict, l_pre_gesture, r_pre_gesture, pre_gesture_dict, annotated_frame
    annotated_frame = np.copy(cv2.cvtColor(output_frame.numpy_view(), cv2.COLOR_RGB2BGR))
    gestures =  result.gestures
    handedness = result.handedness
    multi_hand_landmarks_list = [multi_hand_landmarks for multi_hand_landmarks in result.hand_landmarks]
    
    if gestures:
        for i, (hand, gesture) in enumerate(zip(handedness, gestures)):
            #gesture_list.append((hand[0].category_name, gesture[0].category_name))q
            hand = hand[0].category_name
            gesture = gesture[0].category_name
            if gesture != 'None':
                if hand == 'Left':
                    hand = 'Right'
            #     #r_gesture_dict[timestamp] = gesture
                    pre_gesture_dict['Right'] = gesture
                else:
                    hand = 'Left'
                    pre_gesture_dict['Left'] = gesture
                print(timestamp, hand, gesture)
            #cv2.putText(annotated_frame, f'{gesture}', (0, 60+30*i), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            
"""
    for hand_landmarks in multi_hand_landmarks_list:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])

        solutions.drawing_utils.draw_landmarks(annotated_frame,
                                  hand_landmarks_proto,
                                  solutions.hands.HAND_CONNECTIONS,
                                  solutions.drawing_styles.get_default_hand_landmarks_style(),
                                  solutions.drawing_styles.get_default_hand_connections_style())
 """       
    #     # cv2.imshow('video', annotated_frame)
    #     # cv2.waitKey(1)
    # print('gesture', timestamp, time.time()-t)
    # # annotated_frame = np.copy(cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_RGB2BGR))

                                                
gesture_recognizer_options = vision.GestureRecognizerOptions(base_options=python.BaseOptions(model_asset_path='gesture_recognizer.task'),
                                                                running_mode=vision.RunningMode.LIVE_STREAM,
                                                                num_hands=2,
                                                                min_hand_detection_confidence = 0.5,
                                                                min_hand_presence_confidence = 0.5,
                                                                min_tracking_confidence= 0.5,
                                                                canned_gesture_classifier_options=mp.tasks.components.processors.ClassifierOptions(category_allowlist = CATEGORY_ALLOWLIST),
                                                                #custom_gesture_classifier_options: mp.tasks.components.processors.ClassifierOptions = dataclasses.field(default_factory=_ClassifierOptions),
                                                                result_callback=gesture_recognizer_callback)

gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_recognizer_options)
l_gesture_dict = dict()
r_gesture_dict = dict()

r_pre_gesture = None
l_pre_gesture = None

pre_gesture_dict = {'Left': 'None', 'Right': 'None'}
        
def pose_detector_callback(result, output_frame, timestamp):
    global annotated_frame
    annotated_frame = np.copy(cv2.cvtColor(output_frame.numpy_view(), cv2.COLOR_RGB2BGR))
    height, width = annotated_frame.shape[:2]
    landmark_names = ('l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist')
    
    gesture_recognizer.recognize_async(output_frame, timestamp)

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
        utils.arm_operation(landmark_dict, annotated_frame, obj_dict, pre_gesture_dict)
            
       
            
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x,y=landmark.y, z=landmark.z) for landmark in pose_landmarks])
        solutions.drawing_utils.draw_landmarks(annotated_frame,
                                            pose_landmarks_proto,
                                            solutions.pose.POSE_CONNECTIONS,
                                            solutions.drawing_styles.get_default_pose_landmarks_style())
  
   
  
def main():
    global annotated_frame, gesture_dict
    scale = 1
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
    
    POSE_DETECTOR_MODEL = 'pose_landmarker_models/pose_landmarker_lite.task'
    # POSE_DETECTOR_MODEL = 'pose_landmarker_models/pose_landmarker_full.task'
    # POSE_DETECTOR_MODEL = 'pose_landmarker_models/pose_landmarker_heavy.task'
    
    
    pose_detector_options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=POSE_DETECTOR_MODEL),
                                                         running_mode=vision.RunningMode.LIVE_STREAM,
                                                         result_callback=pose_detector_callback)


    pose_detector =  vision.PoseLandmarker.create_from_options(pose_detector_options)
    
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
