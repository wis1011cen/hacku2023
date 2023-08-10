import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time


# ランドマークの位置に点を打つ
def draw_landmark(frame, landmarker):
    height, width = frame.shape[:2]
    
    for i, landmark in enumerate(landmarker.landmark):
        lm_pos = np.array([int(landmark.x * width), int(landmark.y * height)])
        
            
        if i == 11:                     #左肩
            color = (0, 255, 255)
            lshoulder_pos = lm_pos
            
        elif i == 12:                   #右肩
            color = (255, 255, 0)
            rshoulder_pos = lm_pos
         
        elif i == 13:                   #左肘
            color = (0, 255, 255)
            lelbow_pos = lm_pos
            
        elif i == 14:                   #右肘
            color = (255, 255, 0)
            relbow_pos = lm_pos

        elif i == 15:                   #左手首
            color = (0, 255, 255)
            lwrist_pos = lm_pos
    
            l_visibility = landmark.visibility
            cv2.putText(frame, f'L:{l_visibility:.2f}', (0, 200), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3, 4)
        elif i == 16:                   #右手首
            color = (255, 255, 0)
            rwrist_pos = lm_pos
            
            r_visibility = landmark.visibility
            cv2.putText(frame, f'R:{r_visibility:.2f}', (0, 250), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3, 4)
            
        if i >= 11 and i <= 16:    
            cv2.putText(frame, f'{landmark.z:.1f}', lm_pos, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2, 4)
        else:
            color = (0, 255, 0)
     
        cv2.circle(frame, lm_pos, 5, color, -1)
        

    return lshoulder_pos, lelbow_pos, lwrist_pos, l_visibility, rshoulder_pos, relbow_pos, rwrist_pos, r_visibility


# ランドマーク間を線で結ぶ
def draw_line(frame, landmarker):
    # landmarkの繋がり表示用
    landmark_line_ids = [ 
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),    # 顔
        (11, 12), (11, 23), (12, 24), (23, 24),                                     # 胴
        (11, 13), (13, 15), (15, 21), (15, 17), (17, 19), (19, 15),                 # 右腕
        (12, 14), (14, 16), (16, 22), (16, 18), (18, 20), (20, 16),                 # 左腕
        (23, 25), (25, 27), (27, 29), (29, 31), (31, 27),                           # 右脚
        (24, 26), (26, 28), (28, 30), (30, 32), (32, 28)                            # 左脚
    ]
    height, width = frame.shape[:2]
    for line_id in landmark_line_ids:
        lm = landmarker.landmark[line_id[0]]
        lm_pos1 = (int(lm.x * width), int(lm.y * height))
        
        lm = landmarker.landmark[line_id[1]]
        lm_pos2 = (int(lm.x * width), int(lm.y * height))

        cv2.line(frame, lm_pos1, lm_pos2, (255, 0, 0), 2)
        
def print_result(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))
    print()
    
def get_result(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global gesture_dict
    gestures =  result.gestures
    handedness = result.handedness
    gesture_dict = dict()
    if gestures:
        for hand, gesture in zip(handedness, gestures):
            # print(hand[0].category_name, gesture[0].category_name)
            gesture_dict[hand[0].category_name] = gesture[0].category_name
        # print(gesture_dict)
    return gesture_dict

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=print_result)

gesture_dict = dict()
cap = cv2.VideoCapture(0)
landmarker =  vision.PoseLandmarker.create_from_options(options)

frame_timestamp_ms = 0
pre_t = time.time()
while True:
    t = time.time() - pre_t
    pre_t = time.time()
    
    try:
        fps = 1/t
    except ZeroDivisionError:
        fps = 0
  
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    landmarker.detect_async(mp_image, frame_timestamp_ms)
    
    # i = 0
    # for hand, gesture in gesture_dict.items():
    #     cv2.putText(frame, f'{hand}:{gesture}', (0, 60+30*i), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    #     i += 1
        
    frame_timestamp_ms += 1
    cv2.putText(frame, f'FPS:{fps:.1f}', (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
      break
    
cap.release()
cv2.destroyAllWindows()
