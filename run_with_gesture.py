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
    global pre_gesture_dict, annotated_frame
    #annotated_frame = np.copy(cv2.cvtColor(output_frame.numpy_view(), cv2.COLOR_RGB2BGR))
    gestures =  result.gestures
    handedness = result.handedness
    
    
    if gestures:
        for hand, gesture in zip(handedness, gestures):
            hand = hand[0].category_name
            gesture = gesture[0].category_name
            #if gesture != 'None':
            if hand == 'Left':
                hand = 'Right'
                pre_gesture_dict['Right'] = gesture
            else:
                hand = 'Left'
                pre_gesture_dict['Left'] = gesture
            
    else:
        pre_gesture_dict['Left'] = 'None'
        pre_gesture_dict['Right'] = 'None'
    #print(timestamp, hand, gesture)
                


    # multi_hand_landmarks_list = [multi_hand_landmarks for multi_hand_landmarks in result.hand_landmarks]
    # for hand_landmarks in multi_hand_landmarks_list:
    #     hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    #     hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])

    #     solutions.drawing_utils.draw_landmarks(annotated_frame,
    #                               hand_landmarks_proto,
    #                               solutions.hands.HAND_CONNECTIONS,
    #                               solutions.drawing_styles.get_default_hand_landmarks_style(),
    #                               solutions.drawing_styles.get_default_hand_connections_style())


                                                

        
def pose_detector_callback(result, output_frame, timestamp):
    global annotated_frame

    gesture_recognizer.recognize_async(output_frame, timestamp)
    
    annotated_frame = np.copy(cv2.cvtColor(output_frame.numpy_view(), cv2.COLOR_RGB2BGR))
    
    for name, (x, y, w, h) in appliance_dict.items():
        cv2.putText(annotated_frame, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(annotated_frame, (x, y),(x+w, y+h), (0, 255, 0), 2)
    
    height, width = annotated_frame.shape[:2]
    landmark_names = ('l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist')
    landmark_dict = dict()
    pose_landmarks_list = result.pose_landmarks
    
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        for i , landmark_name in enumerate(landmark_names, 11):
            landmark_cordinate = np.array([int(pose_landmarks[i].x * width), int(pose_landmarks[i].y * height)])
            landmark_dict[landmark_name] = landmark_cordinate
            landmark_dict['l_visibility'] = pose_landmarks[11].visibility
            landmark_dict['r_visibility'] = pose_landmarks[12].visibility
        
        
        utils.arm_operation(landmark_dict, annotated_frame, appliance_dict, pre_gesture_dict)
            
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x,y=landmark.y, z=landmark.z) for landmark in pose_landmarks])
        solutions.drawing_utils.draw_landmarks(annotated_frame,
                                            pose_landmarks_proto,
                                            solutions.pose.POSE_CONNECTIONS,
                                            solutions.drawing_styles.get_default_pose_landmarks_style())
   
  
   
  
def main():
    global annotated_frame, gesture_recognizer, pre_gesture_dict, appliance_dict
    SCALE = 2
    WIDTH = 640*SCALE
    HEIGHT = 360*SCALE
    
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

    pre_gesture_dict = {'Left': 'None', 'Right': 'None'}

    pose_detector =  vision.PoseLandmarker.create_from_options(pose_detector_options)
    
    appliance_dict = {'tv':(0, 200*SCALE, 100*SCALE, 160*SCALE), 'fan':(550*SCALE, 260*SCALE, 90*SCALE, 100*SCALE)}
    l_start_time_dict = dict()
    r_start_time_dict = dict()
    
    for name in appliance_dict.keys():
        l_start_time_dict[name] = 0
        r_start_time_dict[name] = 0
    start_time_dict = {'Left': l_start_time_dict, 'Right': r_start_time_dict}
    utils.load_start_time_dict(start_time_dict)
    
    timestamp = 0
    pre_t = time.time()
   
    ret, frame = cap.read()
    annotated_frame = np.copy(frame)
   
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
        
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        pose_detector.detect_async(mp_frame, timestamp)
        
        
        
        #print('timestamp',timestamp)
        
        #print('FPS', fps)
        #cv2.putText(annotated_frame, f'FPS:{fps:.1f}', (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        # for name, (x, y, w, h) in appliance_dict.items():
        #     cv2.putText(annotated_frame, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        #     cv2.rectangle(annotated_frame, (x,y),(x+w, y+h), (0,0,255), 2)

       
        cv2.imshow('frame', annotated_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        timestamp += 1
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
