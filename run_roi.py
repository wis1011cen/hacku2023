import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import time
import math
import numpy as np
import src.utils as utils

# 11 - left shoulder
# 12 - right shoulder
# 13 - left elbow
# 14 - right elbow
# 15 - left wrist
# 16 - right wrist
# 17 - left pinky
# 18 - right pinky
# 19 - left index
# 20 - right index
# 21 - left thumb
# 22 - right thumb
        
def pose_detector_callback(result, output_frame, timestamp):
    global annotated_frame, rotated_frame, cropped_frame

    annotated_frame = np.copy(cv2.cvtColor(output_frame.numpy_view(), cv2.COLOR_RGB2BGR))
    rotated_frame = np.copy(annotated_frame)
    
    for name, (x, y, w, h) in appliance_dict.items():
        cv2.putText(annotated_frame, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(annotated_frame, (x, y),(x+w, y+h), (0, 255, 0), 2)
    
    height, width = annotated_frame.shape[:2]
    landmark_names = ('l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
                      'l_pinky', 'r_pinky', 'l_index', 'r_index', 'l_thumb', 'r_thumb')
    landmark_dict = dict()
    pose_landmarks_list = result.pose_landmarks
    
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        for i , landmark_name in enumerate(landmark_names, 11):
            landmark_cordinate = np.array([int(pose_landmarks[i].x * width), int(pose_landmarks[i].y * height)])
            landmark_dict[landmark_name] = landmark_cordinate
        # 肩のVisibility
        landmark_dict['l_visibility'] = pose_landmarks[11].visibility
        landmark_dict['r_visibility'] = pose_landmarks[12].visibility
        
        landmark_dict['r_hip'] = np.array([int(pose_landmarks[24].x * width), int(pose_landmarks[24].y * height)])
        #l_hand_center_x = int(pose_landmarks[19].x * width)
        #l_hand_center_y = int(pose_landmarks[19].y * height)
       # print(landmark_dict)
        size = abs(landmark_dict['r_shoulder'][1] - landmark_dict['r_hip'][1])
        #r_hand_center = pose_landmarks[20].x 
        # w1 = abs(landmark_dict['r_thumb'][0] - landmark_dict['r_pinky'][0])
        # w2 = abs(landmark_dict['r_index'][0] - landmark_dict['r_wrist'][0])
        
        # w = max(w1, w2)

        # h1 = abs(landmark_dict['r_thumb'][1] - landmark_dict['r_pinky'][1])
        # h2 = abs(landmark_dict['r_index'][1] - landmark_dict['r_wrist'][1])
        
        # h = max(h1, h2)
        
        # min_x = min(landmark_dict['r_thumb'][0], landmark_dict['r_index'][0], landmark_dict['r_pinky'][0], landmark_dict['r_wrist'][0])
        # min_y = min(landmark_dict['r_thumb'][1], landmark_dict['r_index'][1], landmark_dict['r_pinky'][1], landmark_dict['r_wrist'][1])
        # max_x = max(landmark_dict['r_thumb'][0], landmark_dict['r_index'][0], landmark_dict['r_pinky'][0], landmark_dict['r_wrist'][0])
        # max_y = max(landmark_dict['r_thumb'][1], landmark_dict['r_index'][1], landmark_dict['r_pinky'][1], landmark_dict['r_wrist'][1])
            
        #center = ((min_x + max_x)/2, (min_y + max_y)/2)
        # center = (int(landmark_dict['r_wrist'][0].dtype), int(landmark_dict['r_wrist'][1]))
        # center = (pose_landmarks[])
        
        #center = list(landmark_dict['r_index'])
        # print(type(center))
        #print(type(center[0]))
        # np.array(0, landmark_dict['r_wrist'][1]))
        #print(landmark_dict['r_index'], landmark_dict['r_wrist'], np.array((0, landmark_dict['r_wrist'][1])))
        #angle = utils.calculate_degree(landmark_dict['r_pinky'], landmark_dict['r_wrist'], np.array((0, landmark_dict['r_wrist'][1])))
        angle = utils.calculate_degree(landmark_dict['r_wrist'], landmark_dict['r_shoulder'], np.array((0, landmark_dict['r_shoulder'][1])))

        


        #
        #cropped_frame = np.copy(cv2.cvtColor(output_frame.numpy_view(), cv2.COLOR_RGB2BGR))
        
        #r_cropped_frame = cropped_frame[min_y-5*h : max_y + 5*h, min_x - 5*w : max_x + 5*w]
        #global rotated_frame
        #print(r_cropped_frame.shape)
        #if r_cropped_frame.shape[0] > 0 and r_cropped_frame.shape[1] > 0:
        
        y_w = landmark_dict['r_wrist'][1] 
        y_e = landmark_dict['r_elbow'][1]
        
        if angle < 90:
            if y_w > y_e:
                angle = -angle
        else:
            if y_w > y_e:
                angle = 180 -angle  
            else:
                angle = angle -180
      
                
            #rotation_deg = angle -45
        
        #print(y_w < y_e, angle, rotation_deg)
        #print((landmark_dict['r_index'][0], landmark_dict['r_index'][1]))
        #print(type(landmark_dict['r_index']))     
        center = ([int(pose_landmarks[20].x * width), int(pose_landmarks[20].y * height)])   
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
        rotated_frame = cv2.warpAffine(src=rotated_frame, M=rotate_matrix, dsize=(640*2,360*2))
        
        # sin = math.sin(math.radians(rotation_deg))
        # cos = math.cos(math.radians(rotation_deg))
        #print(sin, cos)
        #min_y = cos * ()
        
        depth = 0.5*abs(pose_landmarks[20].z)
        #print(depth)
        
        min_y = max(int(landmark_dict['r_index'][1] - depth*size) , 0)
        min_x = max(int(landmark_dict['r_index'][0] - depth*size), 0)
        max_y = int(landmark_dict['r_index'][1] + depth*size)
        max_x = int(landmark_dict['r_index'][0] + depth*size)
        #extent = 5 * max(h, w)
        #cropped_frame = rotated_frame[min_y : min_y + extent, min_x : min_x + extent]
        #cropped_frame = np.copy(rotated_frame)
        cropped_frame = rotated_frame[min_y : max_y, min_x : max_x]
        cv2.rectangle(rotated_frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        if cropped_frame.shape[0] > 0 and cropped_frame.shape[1] > 0:
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
            gesture_recognizer.recognize_async(mp_frame, timestamp)
        else:
            cropped_frame = None
        
        
        
        
        
        utils.arm_operation(landmark_dict, annotated_frame, appliance_dict, pre_gesture_dict)
            
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x,y=landmark.y, z=landmark.z) for landmark in pose_landmarks])
        solutions.drawing_utils.draw_landmarks(annotated_frame,
                                            pose_landmarks_proto,
                                            solutions.pose.POSE_CONNECTIONS,
                                            solutions.drawing_styles.get_default_pose_landmarks_style())
        

    
    
def gesture_recognizer_callback(result, output_frame, timestamp):
    global pre_gesture_dict
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
    #print(timestamp, pre_gesture_dict)
                


    # multi_hand_landmarks_list = [multi_hand_landmarks for multi_hand_landmarks in result.hand_landmarks]
    # for hand_landmarks in multi_hand_landmarks_list:
    #     hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    #     hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])

    #     solutions.drawing_utils.draw_landmarks(annotated_frame,
    #                               hand_landmarks_proto,
    #                               solutions.hands.HAND_CONNECTIONS,
    #                               solutions.drawing_styles.get_default_hand_landmarks_style(),
    #                               solutions.drawing_styles.get_default_hand_connections_style())
   
  
   
  
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

    print(f'resolution:{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}x{cap.get(cv2.CAP_PROP_FRAME_WIDTH)}')
    print('FPS:' ,cap.get(cv2.CAP_PROP_FPS))
   
    pose_detector_options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='models/pose_landmarker_lite.task'),
                                                         running_mode=vision.RunningMode.LIVE_STREAM,
                                                         result_callback=pose_detector_callback)
    
    CATEGORY_ALLOWLIST = ['None', 'Thumb_Up', 'Thumb_Down']
    
    gesture_recognizer_options = vision.GestureRecognizerOptions(base_options=python.BaseOptions(model_asset_path='models/gesture_recognizer.task'),
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
    
    global rotated_frame, cropped_frame
    rotated_frame = np.copy(frame)
    cropped_frame = None
   
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
        #gesture_recognizer.recognize_async(mp_frame, timestamp)
        
        
        
        #print('timestamp',timestamp)
        
        #print('FPS', fps)
        #cv2.putText(annotated_frame, f'FPS:{fps:.1f}', (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        # for name, (x, y, w, h) in appliance_dict.items():
        #     cv2.putText(annotated_frame, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        #     cv2.rectangle(annotated_frame, (x,y),(x+w, y+h), (0,0,255), 2)

       
        cv2.imshow('video', cv2.hconcat([annotated_frame, rotated_frame]))
        # cv2.imshow('rotate', rotated_frame)
        # if cropped_frame is not None:
        #     cv2.imshow('cropped', cropped_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        timestamp += 1
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
