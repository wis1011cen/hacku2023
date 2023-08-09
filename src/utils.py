import cv2
import mediapipe as mp
import numpy as np
import time

L_DURATION = 1
L_COOL_TIME = 5
R_DURATION = 1
R_COOL_TIME = 2
DEGREE_THRESHOLD = 120
VISIBILITY_THRESHOLD = 0.6
OPERATION_DISPLAY_TIME = 0.7

start_time = [{'tv': 0, 'fan': 0}, {'tv': 0, 'fan': 0}]    #LR

# ac_mode = 0
operation_time = 0
pre_wrist_pos = np.zeros(2)
pre_name = None
operation_name = None

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence = 0.8, min_tracking_confidence = 0.5)

def detect_pose(frame, objects_pos):
    global start_time, operation_time
    
    # 推論
    frame.flags.writeable = False
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame.flags.writeable = True
    
   
    landmarker = results.pose_landmarks
    
    if not landmarker:
        start_time = [{name : 0 for name in start_time[0].keys()}, {name : 0 for name in start_time[1].keys()}]
    else:
        lshoulder_pos, lelbow_pos, lwrist_pos, l_visibility, rshoulder_pos, relbow_pos, rwrist_pos, r_visibility= draw_landmark(frame, landmarker)
        draw_line(frame, landmarker)
        
        # 左手
        if l_visibility > VISIBILITY_THRESHOLD:
            arm_operation(lshoulder_pos, lelbow_pos, lwrist_pos, 0, objects_pos, frame)
        
        # 右手
        if r_visibility > VISIBILITY_THRESHOLD:
            arm_operation(rshoulder_pos, relbow_pos, rwrist_pos, 1, objects_pos, frame)
        
    if operation_name is not None and (time.time() - operation_time) < OPERATION_DISPLAY_TIME:
        cv2.putText(frame, operation_name, (0, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
       
    
    
     
    return frame


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
        

def arm_operation(pos1, pos2, pos3, is_right_arm, objects_pos, frame):
    global start_time
    vec_a = pos1 - pos2
    vec_b = pos3 - pos2
    degree = calculate_degree(vec_a, vec_b)
        
    if degree > DEGREE_THRESHOLD:
        if is_right_arm == 0:
            l_ir_operation(frame, pos3, objects_pos, vec_b)
        else:
            r_ir_operation(frame, pos3, objects_pos, vec_b)
    else:
        if is_right_arm == 0:        # 左腕
            start_time[0] = {name : 0 for name in start_time[0].keys()}
        else:                     # 右腕
            start_time[1] = {name : 0 for name in start_time[0].keys()}
        
        
    #cv2.putText(frame, f'{degree:.0f}', (0, 0), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        
        
# 3点のなす角度を求める
def calculate_degree(vec_a, vec_b):        
    len_vec_a = np.linalg.norm(vec_a)
    len_vec_b = np.linalg.norm(vec_b)
    
    cos = np.inner(vec_a, vec_b)/(len_vec_a * len_vec_b)
    rad = np.arccos(cos)
    return  np.rad2deg(rad)

#from ir.irrp import ir_lightning
    
# 左腕で電源ON
def l_ir_operation(frame, wrist_pos, obj_dict, vec_b):
    global start_time, operation_time
    ex_pos = wrist_pos + 20 * vec_b    #腕を伸ばした先
    color = (255, 0, 0)
    
    if (time.time() - operation_time) > L_COOL_TIME or operation_time == 0:
        for name, obj_pos in obj_dict.items():
            color = (255, 0, 0)
            
            #線分が交わるか判定
            if hit_detection(wrist_pos, ex_pos, obj_pos):
                if start_time[0][name] == 0:
                    start_time[0][name] = time.time()
                else:
                    duration_time = time.time() - start_time[0][name] 
                    cv2.putText(frame, f'L:{duration_time:.1f}', (0, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                 
                
                    # 一定時間以上交わったとき
                    if duration_time > L_DURATION: 
                        end = 'on'
                        # if name == 'ac':
                        #     if ac_mode == 0:
                        #         ac_mode = 1
                        #     else:
                        #         end = 'off'
                        #         ac_mode = 0
                        ir_operation(name, end, 0)
                        
                        duration_time = 0
                            
                color = (0, 0, 255)
                break
            else:
                start_time[0][name] = 0

            
    cv2.line(frame, wrist_pos, ex_pos, color, 2)
    
# 右腕:チャンネル変更 
def r_ir_operation(frame, wrist_pos, obj_dict, vec_b):
    global start_time, operation_time, ac_mode, pre_wrist_pos, pre_name
    ex_pos = wrist_pos + 20 * vec_b    #腕を伸ばした先
    color = (255, 0, 0)
    if (time.time() - operation_time) > R_COOL_TIME or operation_time == 0:
        for name, obj_pos in obj_dict.items():
            color = (255, 0, 0)
            # 線分が交わるか判定
            if hit_detection(wrist_pos, ex_pos, obj_pos):
                if start_time[1][name] == 0:
                    start_time[1][name] = time.time()
                else:
                    duration_time = time.time() - start_time[1][name]
                    cv2.putText(frame, f'R:{duration_time:.1f}', (0, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    
                    # 一定時間以上交わったときの手首の位置と操作対象を記録
                    if duration_time > R_DURATION:  
                        pre_wrist_pos = wrist_pos
                        pre_name = name      
                color = (0, 0, 255)
                break
                    
            else:
                start_time[1][name] = 0
                
    # 線分が交わった状態から離れた場合
    if isinstance(pre_name, str):
        obj_pos = obj_dict[pre_name]
        
        if not hit_detection(wrist_pos, ex_pos, obj_pos):
            end = 'up' if wrist_pos[1] < pre_wrist_pos[1] else 'down'
            
            ir_operation(pre_name, end, 1)
            
            pre_name = None
            
    cv2.line(frame, wrist_pos, ex_pos, color, 2)
  
  

def ir_operation(name, end, is_right):
    global operation_time, operation_name
    print('lightning')
    #name = f'{name}-{end}'
    operation_name = f'{name}-{end}'
    
    # あとで作る
    #ir_lightning()
    
    operation_time = time.time()
    
    start_time[is_right][name] = 0
                        
    
# 操作対象と線の当たり判定
def hit_detection(pos1, pos2, obj_pos):
    x, y, w, h = obj_pos
    ul_pos = np.array([x, y])       #左上
    ur_pos = np.array([x+w, y])     #右上
    lr_pos = np.array([x+w, y+h])   #右下
    ll_pos = np.array([x, y+h])     #左下

    left_edge_TF: bool = cross_detection(pos1, pos2, ul_pos, ll_pos)
    right_edge_TF: bool = cross_detection(pos1, pos2, ur_pos, lr_pos)
    top_edge_TF: bool = cross_detection(pos1, pos2, ul_pos, ur_pos)
    bottom_edge_TF: bool = cross_detection(pos1, pos2, ll_pos, lr_pos)
    
    if left_edge_TF or right_edge_TF or top_edge_TF or bottom_edge_TF:
        return True
    else:
        return False
    
# 線分と線分の当たり判定
def cross_detection(pos_a, pos_b, pos_c, pos_d):
    vec_ab = pos_b - pos_a
    vec_ac = pos_c - pos_a
    vec_ad = pos_d - pos_a
    product1 = np.cross(vec_ab, vec_ac) * np.cross(vec_ab, vec_ad)
        
    vec_cd = pos_d - pos_c
    vec_cb = pos_b - pos_c
    vec_ca = - vec_ac
    product2 = np.cross(vec_cd, vec_cb) * np.cross(vec_cd, vec_ca)
    
    if product1 < 0 and product2 < 0:
        return True
    else:
        return False
    
    
def test():
    WIDTH = 640*2
    HEIGHT = 360*2

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    #width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  
    print(f'FPS:{cap.get(cv2.CAP_PROP_FPS)}')
    print(f'resolution:{cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')
    
    # x,y,w,h
    obj_dict = {'tv':(0, 400, 280, 200), 'fan':(1000, 0, 200, 100)}
    tv_pos, fan_pos = obj_dict.values()
    
    tv_x, tv_y, tv_w, tv_h = tv_pos
    fan_x, fan_y, fan_w, fan_h = fan_pos 
    
    
    if not cap.isOpened():
        print('Cannot open a camera')
        return 1
    
    pre_t = time.time()
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
       
        frame = detect_pose(frame, obj_dict)
        
        cv2.rectangle(frame, (tv_x, tv_y), (tv_x + tv_w, tv_y + tv_h), (0, 0, 255), 2)
        cv2.rectangle(frame, (fan_x, fan_y), (fan_x + fan_w, fan_y + fan_h), (0, 0, 255), 2)
        cv2.putText(frame, f'FPS:{fps:.1f}', (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('video', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
    cv2.destroyAllWindows()
    cap.release()
        

    
if __name__ == "__main__":
    test()
    

    
    