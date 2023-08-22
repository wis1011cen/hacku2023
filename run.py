import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import time
import numpy as np
import argparse
import src.utils as utils
import src.roi as roi
import csv

# 11 - left shoulder
# 12 - right shoulder
# 13 - left elbow
# 14 - right elbow
# 15 - left wrist
# 16 - right wrist

def pose_detector_callback(result, output_frame, timestamp):
    global annotated_frame
    annotated_frame = np.copy(cv2.cvtColor(output_frame.numpy_view(), cv2.COLOR_RGB2BGR))
    height, width = annotated_frame.shape[:2]
    landmark_names = ('l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist')
    landmark_dict = dict()
    pose_landmarks_list = result.pose_landmarks
    
    for appliance_name, (x, y, w, h) in appliance_dict.items():
        cv2.putText(annotated_frame, appliance_name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(annotated_frame, (x, y),(x+w, y+h), (0, 255, 0), 2)
    
   # if pose_landmarks_list:
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        for i , landmark_name in enumerate(landmark_names, 11):
            landmark_cordinate = np.array([int(pose_landmarks[i].x * width), int(pose_landmarks[i].y * height)])
            landmark_dict[landmark_name] = landmark_cordinate
            
            # 11:
            #landmark_dict['l_visibility'] = min(pose_landmarks[11].visibility, pose_landmarks[13].visibility, pose_landmarks[15].visibility,)
            #landmark_dict['r_visibility'] = min(pose_landmarks[12].visibility, pose_landmarks[14].visibility, pose_landmarks[16].visibility,)
            landmark_dict['l_visibility'] = pose_landmarks[15].visibility
            landmark_dict['r_visibility'] = pose_landmarks[16].visibility
        
        utils.arm_operation(landmark_dict, annotated_frame, appliance_dict)
            
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x,y=landmark.y, z=landmark.z) for landmark in pose_landmarks])
        solutions.drawing_utils.draw_landmarks(annotated_frame,
                                            pose_landmarks_proto,
                                            solutions.pose.POSE_CONNECTIONS,
                                            solutions.drawing_styles.get_default_pose_landmarks_style())
        
            
def main():
    global annotated_frame, appliance_dict, start_time_dict
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', action='store_true')
    #parser.add_argument('--csv', action='store_true') 
    args = parser.parse_args()
    
    SCALE = 2
    WIDTH = 640*SCALE
    HEIGHT = 360*SCALE
    
    
    
    #codec = cv2.VideoWriter_fourcc(*'mp4v')
    #video = cv2.VideoWriter('video.mp4', codec, 10, (WIDTH*2, HEIGHT))
    #video2 = cv2.VideoWriter('video2.mp4', codec, 10, (WIDTH, HEIGHT))
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open a video capture.")
        exit(-1)
        
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        
    # height = int())
    # width = int()


    print(f'resolution:{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}x{cap.get(cv2.CAP_PROP_FRAME_WIDTH)}')
    print('FPS:' ,cap.get(cv2.CAP_PROP_FPS))
    
    pose_detector_options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='models/pose_landmarker_lite.task'),
                                                         running_mode=vision.RunningMode.LIVE_STREAM,
                                                         result_callback=pose_detector_callback)

    pose_detector =  vision.PoseLandmarker.create_from_options(pose_detector_options)
    
    # 自分で指定する場合
    if args.set:
        appliance_dict = roi.select_roi(cap, 'video')
        appliance_list = []
        for name, (x, y, w, h) in appliance_dict.items():
            appliance_list.append([name, x, y, w, h])
            
        with open('src/roi.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(appliance_list)
            
    # csvファイルから読み込む場合 
    else:
        appliance_dict = dict()
        with open('src/roi.csv') as f:
            reader = csv.reader(f)
            for row in reader:
                name = row[0]
                x, y, w, h = map(int, row[1:])
                appliance_dict[name] = (x, y, w, h)
                
    #else:
        #appliance_dict = {'fan':(0, 200*SCALE, 100*SCALE, 160*SCALE), 'tv':(550*SCALE, 260*SCALE, 90*SCALE, 100*SCALE)}
    
    l_start_time_dict = dict()
    r_start_time_dict = dict()
    for name in appliance_dict.keys():
        l_start_time_dict[name] = 0
        r_start_time_dict[name] = 0
    start_time_dict = {'Left': l_start_time_dict, 'Right': r_start_time_dict}
    #print(start_time_dict)
    
    utils.load_start_time_dict(start_time_dict)
    
    timestamp = 0
    # pre_t = time.time()
  
    ret, frame = cap.read()
    annotated_frame = np.copy(frame)
    
    while True:
        # t = time.time() - pre_t
        # pre_t = time.time()
       
        
        # try:
        #     fps = 1/t
        # except ZeroDivisionError:
        #     fps = 0
    
        ret, frame = cap.read()
        
        if not ret:
            print('error')
            break
        
        #frame = cv2.flip(frame, 1)
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        pose_detector.detect_async(mp_frame, timestamp)
        
        
        #print('FPS', fps)
        #cv2.putText(annotated_frame, f'FPS:{fps:.1f}', (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        # for name, (x, y, w, h) in appliance_dict.items():
        #     cv2.putText(annotated_frame, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        #     cv2.rectangle(annotated_frame, (x, y),(x+w, y+h), (0, 255, 0), 2)
        
        #concat = cv2.hconcat([annotated_frame, frame])
        #video.write(concat)
        #video2.write(frame)
        cv2.imshow('Video', annotated_frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        timestamp += 1
        
    cap.release()
    cv2.destroyAllWindows()
    video.release()
    #video2.release()
    
if __name__ == '__main__':
    main()
