import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import time
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
# 23 - left hip
# 24 - right hip

def l_rotate_and_crop(landmark_dict, input_frame):
    RATIO = 0.5
    angle = utils.calculate_degree(landmark_dict['l_wrist'], landmark_dict['l_shoulder'], np.array((0, landmark_dict['l_shoulder'][1])))
    y_wrist = landmark_dict['l_wrist'][1] 
    y_elbow = landmark_dict['l_elbow'][1]

    if angle < 90:
        if y_wrist > y_elbow:
            angle = - 90 - angle
        else:
            angle = angle - 90
    else:
        if y_wrist > y_elbow:
            angle = 270 -angle  
        else:
            angle = angle -90

    center = (int(landmark_dict['l_index'][0]), int(landmark_dict['l_index'][1]))
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    l_rotated_frame = cv2.warpAffine(src=input_frame, M=rotate_matrix, dsize=(640*2,360*2))
    
    size = abs(landmark_dict['l_shoulder'][1] - landmark_dict['l_hip'][1])
    
    min_y = max(int(landmark_dict['l_index'][1] - RATIO*size) , 0)
    min_x = max(int(landmark_dict['l_index'][0] - RATIO*size), 0)
    max_y = int(landmark_dict['l_index'][1] + RATIO*size)
    max_x = int(landmark_dict['l_index'][0] + RATIO*size)
    
    l_cropped_frame = l_rotated_frame[min_y : max_y, min_x : max_x]
    cv2.rectangle(l_rotated_frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    
    if not(l_cropped_frame.shape[0] > 0 and l_cropped_frame.shape[1] > 0):
        l_cropped_frame = None
    
    return l_rotated_frame, l_cropped_frame
    


def r_rotate_and_crop(landmark_dict, input_frame):
    RATIO = 0.5
    angle = utils.calculate_degree(landmark_dict['r_wrist'], landmark_dict['r_shoulder'], np.array((0, landmark_dict['r_shoulder'][1])))
    y_wrist = landmark_dict['r_wrist'][1] 
    y_elbow = landmark_dict['r_elbow'][1]

    if angle < 90:
        if y_wrist > y_elbow:
            angle = -angle
    else:
        if y_wrist > y_elbow:
            angle = 180 -angle  
        else:
            angle = angle -180

    center = (int(landmark_dict['r_index'][0]), int(landmark_dict['r_index'][1]))
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    r_rotated_frame = cv2.warpAffine(src=input_frame, M=rotate_matrix, dsize=(640*2,360*2))
    
    size = abs(landmark_dict['r_shoulder'][1] - landmark_dict['r_hip'][1])
    
    min_y = max(int(landmark_dict['r_index'][1] - RATIO*size) , 0)
    min_x = max(int(landmark_dict['r_index'][0] - RATIO*size), 0)
    max_y = int(landmark_dict['r_index'][1] + RATIO*size)
    max_x = int(landmark_dict['r_index'][0] + RATIO*size)
    
    r_cropped_frame = r_rotated_frame[min_y : max_y, min_x : max_x]
    cv2.rectangle(r_rotated_frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    
    if not(r_cropped_frame.shape[0] > 0 and r_cropped_frame.shape[1] > 0):
        r_cropped_frame = None
        

    return r_rotated_frame, r_cropped_frame 

        
def pose_detector_callback(result, output_frame, timestamp):
    global annotated_frame, rotated_frame, cropped_frame

    annotated_frame = np.copy(cv2.cvtColor(output_frame.numpy_view(), cv2.COLOR_RGB2BGR))
    copy_frame = np.copy(annotated_frame)
    
    for name, (x, y, w, h) in appliance_dict.items():
        cv2.putText(annotated_frame, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(annotated_frame, (x, y),(x+w, y+h), (0, 255, 0), 2)
    
    height, width = annotated_frame.shape[:2]
    landmark_names = ('l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
                      'l_pinky', 'r_pinky', 'l_index', 'r_index', 'l_thumb', 'r_thumb', 'l_hip', 'r_hip')
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
        
        global rotated_frame
        l_rotated_frame, l_cropped_frame = l_rotate_and_crop(landmark_dict, copy_frame)
        r_rotated_frame, r_cropped_frame= r_rotate_and_crop(landmark_dict, copy_frame)
        rotated_frame = cv2.hconcat([r_rotated_frame, l_rotated_frame])
        if l_cropped_frame is not None and r_cropped_frame is not None:
            # print('L', l_cropped_frame.shape)
            # print('R', r_cropped_frame.shape)
    
            diff = l_cropped_frame.shape[0] - r_cropped_frame.shape[0]
            # print(diff)
            if diff > 0:
                l_padding = cv2.copyMakeBorder(l_cropped_frame, 0, 0, 0, 0, cv2.BORDER_CONSTANT, (0,0,0))
                r_padding = cv2.copyMakeBorder(r_cropped_frame, 0, diff, 0, 0, cv2.BORDER_CONSTANT, (0,0,0))
            else:
                diff = - diff
                l_padding = cv2.copyMakeBorder(l_cropped_frame, 0, diff, 0, 0, cv2.BORDER_CONSTANT, (0,0,0))
                r_padding = cv2.copyMakeBorder(r_cropped_frame, 0, 0, 0, 0, cv2.BORDER_CONSTANT, (0,0,0))
            # print(l_padding.shape)
            # print(r_padding.shape)
            # l_resize = cv2.resize(l_cropped_frame, (l_cropped_frame.shape[1], h_min))
            # r_resize = cv2.resize(r_cropped_frame, (r_cropped_frame.shape[1], h_min))
            
            cropped_frame = cv2.hconcat([r_padding, l_padding])
       
        elif l_cropped_frame is not None:
            cropped_frame = r_cropped_frame
        
        elif r_cropped_frame is not None:
            cropped_frame = l_cropped_frame
        else:
            cropped_frame = None
        if cropped_frame is not None:
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
            gesture_recognizer.recognize_async(mp_frame, timestamp)

        utils.arm_operation(landmark_dict, annotated_frame, appliance_dict, pre_gesture_dict)
            
           
        
           
        # Pose Landmark の描画 
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
    
    CATEGORY_ALLOWLIST = ["None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"]
    
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
        
        cv2.imshow('main', annotated_frame)
        # cv2.imshow('video', cv2.hconcat([annotated_frame, rotated_frame]))
        # cv2.imshow('rotated', rotated_frame)
       
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
