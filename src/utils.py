import cv2
import mediapipe as mp
import numpy as np
import time
# import src.ssh_request as ssh_request

L_DURATION = 0.5
L_COOL_TIME = 1
R_DURATION = 1
R_COOL_TIME = 2
DEGREE_THRESHOLD = 120
VISIBILITY_THRESHOLD = 0.6

start_time = [{'tv1': 0, 'tv2': 0},{'tv1': 0, 'tv2': 0}]    #LR

ac_mode = 0
ssh_time = 0
pre_wrist_pos = np.zeros(2)
pre_name = None



mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence = 0.8, min_tracking_confidence = 0.5)

def detect_pose(frame, depth_map, objects_pos, client):
    global start_time
    
    # 推論
    frame.flags.writeable = False
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame.flags.writeable = True
    
   
    landmarker = results.pose_landmarks
    #print(start_time)
    
    if not landmarker:
        start_time = [{name : 0 for name in start_time[0].keys()}, {name : 0 for name in start_time[1].keys()}]
    else:
        lshoulder_pos, lelbow_pos, lwrist_pos, l_visibility, rshoulder_pos, relbow_pos, rwrist_pos, r_visibility= draw_landmark(frame, depth_map, landmarker)
        draw_line(frame, landmarker)
        
        # 左手
        if l_visibility > VISIBILITY_THRESHOLD:
            arm_operation(lshoulder_pos, lelbow_pos, lwrist_pos, 0, objects_pos, frame, client)
        
        # 右手
        if r_visibility > VISIBILITY_THRESHOLD:
            arm_operation(rshoulder_pos, relbow_pos, rwrist_pos, 1, objects_pos, frame, client)
     
    return frame


# ランドマークの位置に点を打つ
def draw_landmark(frame, depth_map, landmarker):
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
            cv2.putText(frame, f'L:{l_visibility:.2f}', (500, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3, 4)
        elif i == 16:                   #右手首
            color = (255, 255, 0)
            rwrist_pos = lm_pos
            r_visibility = landmark.visibility
            cv2.putText(frame, f'R:{r_visibility:.2f}', (500, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3, 4)
        else:
            color = (0, 255, 0)
            
        try:
            depth = depth_map[lm_pos[1], lm_pos[0]]
            cv2.putText(frame, f'{depth}', lm_pos, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2, 4)
            
        except (IndexError, TypeError):
            pass
           
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
        

def arm_operation(pos1, pos2, pos3, is_right_arm, objects_pos, frame, client):
    global start_time
    vec_a = pos1 - pos2
    vec_b = pos3 - pos2
    degree = calculate_degree(vec_a, vec_b)
        
    if degree > DEGREE_THRESHOLD:
        if is_right_arm == 0:
            l_ir_operation(frame, pos3, objects_pos, vec_b, client)
        else:
            r_ir_operation(frame, pos3, objects_pos, vec_b, client)
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


    
# 左腕で電源ON
def l_ir_operation(frame, wrist_pos, obj_dict, vec_b, client):
    global start_time, ssh_time, ac_mode
    ex_pos = wrist_pos + 20 * vec_b    #腕を伸ばした先
    color = (255, 0, 0)
    #print(pre_wrist_pos)
    if (time.time() - ssh_time) > L_COOL_TIME or ssh_time == 0:
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
                        if name == 'ac':
                            if ac_mode == 0:
                                ac_mode = 1
                            else:
                                end = 'off'
                                ac_mode = 0
                        # if client is not None:
                            send_ssh(frame, name, end, 0, client)
                        
                        duration_time = 0
                            
                color = (0, 0, 255)
                break
            else:
                start_time[0][name] = 0

            
    cv2.line(frame, wrist_pos, ex_pos, color, 2)
    
# 右腕:チャンネル変更 
def r_ir_operation(frame, wrist_pos, obj_dict, vec_b, client):
    global start_time, ssh_time, ac_mode, pre_wrist_pos, pre_name
    ex_pos = wrist_pos + 20 * vec_b    #腕を伸ばした先
    color = (255, 0, 0)
    if (time.time() - ssh_time) > R_COOL_TIME or ssh_time == 0:
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
            
            send_ssh(frame, pre_name, end, 1, client)
            pre_name = None
            
    cv2.line(frame, wrist_pos, ex_pos, color, 2)
 
def send_ssh(frame, name, end, is_right, client):
    global ssh_time
    name = f'{name}-{end}'
    cv2.putText(frame, name, (0, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    
    # すぐに表示
    cv2.imshow('video', frame)
    cv2.waitKey(1)
    
    t = time.time()
    if client is not None:
        ssh_request.ir_lighting(client, name)
    else:
        print('ssh is not connected.')
    ssh_time = time.time()
    print(f'ssh time:{time.time()-t}')
    
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
    with ssh_request.connect() as client:

        cap = cv2.VideoCapture(0)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(f'FPS:{fps}')
        print(f'resolution:{width}x{height}')
        
        # x,y,w,h
        obj_dict = {'tv':(1000, 400, 280, 200), 'ac':(200, 0, 200, 100)}
        tv_pos, ac_pos = obj_dict.values()
        
        tv_x, tv_y, tv_w, tv_h = tv_pos
        ac_x, ac_y, ac_w, ac_h = ac_pos 

        if cap.isOpened():
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                
                frame = cv2.flip(frame, 1)
                frame = detect_pose(frame, obj_dict, client)
                
                cv2.rectangle(frame, (tv_x, tv_y), (tv_x + tv_w, tv_y + tv_h), (0, 0, 255), 2)
                cv2.rectangle(frame, (ac_x, ac_y), (ac_x + ac_w, ac_y + ac_h), (0, 0, 255), 2)
                
                
                cv2.imshow('video', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 0x1b:
                    break
                    
            
            cap.release()

def test_without_ssh():
    # import stereo_camera.stereo as stereo
    import stereo
    
    L_CAMERA_DEVISE = 1
    R_CAMERA_DEVICE = 3
    WIDTH = 640
    HEIGHT = 480
    # cap = cv2.VideoCapture(0)
    capl = cv2.VideoCapture(L_CAMERA_DEVISE)
    capr = cv2.VideoCapture(R_CAMERA_DEVICE)

    capl.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    capl.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    capr.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    capr.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    capl.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    capr.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    capl.set(cv2.CAP_PROP_EXPOSURE, -5)
    capr.set(cv2.CAP_PROP_EXPOSURE, -5)
    
    fps = int(capl.get(cv2.CAP_PROP_FPS))
    
    print(f'FPS:{fps}')
    print(f'resolution:{WIDTH}x{HEIGHT}')
    
    # x,y,w,h
    obj_dict = {'tv':(1000, 400, 280, 200), 'ac':(200, 0, 200, 100)}
    tv_pos, ac_pos = obj_dict.values()
    
    tv_x, tv_y, tv_w, tv_h = tv_pos
    ac_x, ac_y, ac_w, ac_h = ac_pos 
    
    map1_l, map2_l, map1_r, map2_r, wls_filter, left_matcher, right_matcher = stereo.load_caliblation_data(WIDTH, HEIGHT)

    if capl.isOpened():
        while True:
            retl, framel = capl.read()
            retr, framer = capr.read()
            
            if not retl or not retr:
                continue
            
            framel_gray, framer_gray, depth_map = stereo.depth_estimate(framel, framer, map1_l, map2_l, map1_r, map2_r, wls_filter, left_matcher, right_matcher)
            
            # print('max', np.max(filtered_frame) , 'min', np.min(filtered_frame))
            # max_idx = np.argmax(filtered_frame)
            # min_idx = np.argmin(filtered_frame)
            max_idx = np.unravel_index(np.argmax(depth_map), depth_map.shape)
            min_idx = np.unravel_index(np.argmin(depth_map), depth_map.shape)
            # print(idx)
            # print(max_idx, min_idx)
            
            THRESHOLD = -1000
            filtered_frame = np.where(depth_map < THRESHOLD, THRESHOLD, depth_map)
            filtered_frame = np.where(filtered_frame > 1000, 1000, filtered_frame)
            
            filtered_frame = cv2.normalize(src=filtered_frame, dst=filtered_frame, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
            filtered_frame = np.uint8(filtered_frame)
            # print(filtered_frame.)
            # print('max_', filtered_frame[max_idx], 'min_', filtered_frame[min_idx])
            # cv2.imshow("Disparity", filtered_frame)
            # framel = cv2.flip(framel, 1)
            framel = detect_pose(framel, depth_map, obj_dict, client = None)
            
            cv2.rectangle(framel, (tv_x, tv_y), (tv_x + tv_w, tv_y + tv_h), (0, 0, 255), 2)
            cv2.rectangle(framel, (ac_x, ac_y), (ac_x + ac_w, ac_y + ac_h), (0, 0, 255), 2)
            
            # cv2.imshow('video', framel)
            filtered_frame = np.stack((filtered_frame,)*3, -1)
            cv2.imshow('depth', cv2.hconcat([framel, filtered_frame]))
            # cv2.imshow('stereo', cv2.hconcat([framel_gray, framer_gray, filtered_frame]))
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 0x1b:
                break
                
        
        capl.release()
        
def test_single_camera():
    WIDTH = 640
    HEIGHT = 480

    cap = cv2.VideoCapture(0)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f'FPS:{fps}')
    print(f'resolution:{WIDTH}x{HEIGHT}')
    
    # x,y,w,h
    obj_dict = {'tv':(1000, 400, 280, 200), 'ac':(200, 0, 200, 100)}
    tv_pos, ac_pos = obj_dict.values()
    
    tv_x, tv_y, tv_w, tv_h = tv_pos
    ac_x, ac_y, ac_w, ac_h = ac_pos 
    
    
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            
            if not ret:
                continue
           
            frame = detect_pose(frame, None, obj_dict, client = None)
            
            cv2.rectangle(frame, (tv_x, tv_y), (tv_x + tv_w, tv_y + tv_h), (0, 0, 255), 2)
            cv2.rectangle(frame, (ac_x, ac_y), (ac_x + ac_w, ac_y + ac_h), (0, 0, 255), 2)
            
            cv2.imshow('video', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 0x1b:
                break
                
        
        cap.release()
        

    
if __name__ == "__main__":
    # import ssh_request
    # import stereo_camera.stereo
    test_without_ssh()
    #test_single_camera()
    

    
    