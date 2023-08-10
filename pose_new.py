import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import numpy as np



landmark_line_ids = [ 
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),    # 顔
    (11, 12), (11, 23), (12, 24), (23, 24),                                     # 胴
    (11, 13), (13, 15), (15, 21), (15, 17), (17, 19), (19, 15),                 # 右腕
    (12, 14), (14, 16), (16, 22), (16, 18), (18, 20), (20, 16),                 # 左腕
    (23, 25), (25, 27), (27, 29), (29, 31), (31, 27),                           # 右脚
    (24, 26), (26, 28), (28, 30), (30, 32), (32, 28)                            # 左脚
]
# ランドマークの位置に点を打つ
def draw_landmark(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global frame, landmark_dict, line_dict
    landmark_dict = dict()
    line_dict = dict()
    height, width = frame.shape[:2]
    # print(result.pose_landmarks[0])
    for i, landmark in enumerate(result.pose_landmarks[0]):
        #print(i, landmark.x, landmark.y, landmark.z)
        lm_pos = np.array([int(landmark.x * width), int(landmark.y * height)])
        # print(i, lm_pos)
        landmark_dict[i] =  lm_pos
            

    #height, width = frame.shape[:2]
    for line_id in landmark_line_ids:
        lm = result.pose_landmarks[0][line_id[0]]
        lm_pos1 = (int(lm.x * width), int(lm.y * height))
        
        
        lm = result.pose_landmarks[0][line_id[1]]
        lm_pos2 = (int(lm.x * width), int(lm.y * height))
        # print(line_id, lm_pos1, lm_pos2)
        line_dict[line_id] = (lm_pos1, lm_pos2)


options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=draw_landmark)

landmark_dict = dict()
line_dict = dict()
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
    # print(line_dict)
    
    for pos in landmark_dict.values():
        cv2.circle(frame, pos, 5, (255, 0, 0), -1)
    for lm_pos in line_dict.values():
        cv2.line(frame, lm_pos[0], lm_pos[1], (255, 0, 0), 2)

    frame_timestamp_ms += 1

    cv2.putText(frame, f'FPS:{fps:.1f}', (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
      break
    
cap.release()
cv2.destroyAllWindows()
