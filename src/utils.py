import cv2
import numpy as np
import time
import ir.irrp as irrp

L_DURATION = 0.5
L_COOL_TIME = 2
R_DURATION = 0.5
R_COOL_TIME = 2
DEGREE_THRESHOLD = 120
VISIBILITY_THRESHOLD = 0.5
OPERATION_DISPLAY_TIME = 0.7
DIFF_THRESHOLD = 50
MIN_DEGREE = 45
MAX_DEGREE = 135

operation_time = 0
pre_wrist_pos = np.zeros(2)
pre_appliance_name = None
operation_name = None

def load_start_time_dict(input_start_time_dict):
    global start_time_dict
    start_time_dict = input_start_time_dict
    
 
def arm_operation(landmark_dict, annotated_frame, appliance_dict):
    global start_time_dict, pre_appliance_name 
        
    if operation_name is not None and (time.time() - operation_time) < OPERATION_DISPLAY_TIME:
        x, y, w, h = appliance_dict[operation_name.split('-')[0]]
        cv2.putText(annotated_frame, operation_name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(annotated_frame, (x, y),(x+w, y+h), (0, 0, 255), 2)
        
    if landmark_dict['l_visibility'] > VISIBILITY_THRESHOLD:
        degree1 = calculate_degree(landmark_dict['l_shoulder'], landmark_dict['l_elbow'], landmark_dict['l_wrist'])
        degree2 = calculate_degree(landmark_dict['l_elbow'], landmark_dict['l_wrist'], np.array([0, landmark_dict['l_wrist'][1]]))
        if degree1 > DEGREE_THRESHOLD:
            #print(degree2)
            if degree2 < MIN_DEGREE or degree2 > MAX_DEGREE:
                l_ir_operation(annotated_frame, landmark_dict['l_wrist'], appliance_dict, landmark_dict['l_elbow'])
        else:
            start_time_dict['Left'] = {name : 0 for name in start_time_dict['Left'].keys()}
   
    # if landmark_dict['r_visibility'] > VISIBILITY_THRESHOLD:
    #     degree1 = calculate_degree(landmark_dict['r_shoulder'], landmark_dict['r_elbow'], landmark_dict['r_wrist'])
    #     #degree2 = calculate_degree(landmark_dict['r_elbow'], landmark_dict['r_wrist'], np.array([0, landmark_dict['r_wrist'][1]]))

    #     if degree1 > DEGREE_THRESHOLD:
    #         #if degree2 < 45 or degree2 > 135:
    #         r_ir_operation(annotated_frame, landmark_dict['r_wrist'], appliance_dict, landmark_dict['r_elbow'])
    #     else:
    #          start_time_dict['Right'] = {name : 0 for name in start_time_dict['Right'].keys()}
    #          #print(start_time_dict)
    
    degree1 = calculate_degree(landmark_dict['r_shoulder'], landmark_dict['r_elbow'], landmark_dict['r_wrist'])
    degree2 = calculate_degree(landmark_dict['r_elbow'], landmark_dict['r_wrist'], np.array([0, landmark_dict['r_wrist'][1]]))

        # degree = calculate_degree(elbow_pos, wrist_pos, np.array([0, wrist_pos[1]]))
    cv2.putText(annotated_frame,  f"L:{landmark_dict['l_visibility']:.2f}", (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(annotated_frame,  f"R:{landmark_dict['r_visibility']:.2f}", (0, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    if degree1 > DEGREE_THRESHOLD and (degree2 < MIN_DEGREE or degree2 > MAX_DEGREE) and landmark_dict['r_visibility'] > VISIBILITY_THRESHOLD:


        #if degree1 > DEGREE_THRESHOLD:
            #if degree2 < 45 or degree2 > 135:
        r_ir_operation(annotated_frame, landmark_dict['r_wrist'], appliance_dict, landmark_dict['r_elbow'])
    
    else:
        start_time_dict['Right'] = {name : 0 for name in start_time_dict['Right'].keys()}
             #print(start_time_dict)
        
    if pre_appliance_name is not None:
        appliance_pos = appliance_dict[pre_appliance_name]
        ex_pos = landmark_dict['r_wrist'] + 20 * (landmark_dict['r_wrist'] - landmark_dict['r_elbow'])    #腕を伸ばした先
        diff = abs(landmark_dict['r_wrist'][1]- pre_wrist_pos[1])
        # print(diff)
        if not hit_detection(landmark_dict['r_wrist'], ex_pos, appliance_pos) and diff > DIFF_THRESHOLD:
            if landmark_dict['r_wrist'][1] < pre_wrist_pos[1]:
                end = 'up'
            else:
                end = 'down'
            
            ir_operation(pre_appliance_name, end, 'Right')
            
            pre_appliance_name = None
    
         

        
        
# 3点のなす角度を求める
def calculate_degree(pos1, pos2, pos3):  
    vec_a = pos1 - pos2
    vec_b = pos3 - pos2      
    len_vec_a = np.linalg.norm(vec_a)
    len_vec_b = np.linalg.norm(vec_b)
    
    cos = np.inner(vec_a, vec_b)/(len_vec_a * len_vec_b)
    deg = np.rad2deg(np.arccos(cos))
    return deg

    
# 左腕で電源ON
def l_ir_operation(annotated_frame, wrist_pos, appliance_dict, elbow_pos):
    global operation_time, start_time_dict
    extended_pos = wrist_pos + 20 * (wrist_pos - elbow_pos)    #腕を伸ばした先
    color = (255, 0, 0)
    
    if (time.time() - operation_time) > L_COOL_TIME or operation_time == 0:
        for appliance_name, appliance_pos in appliance_dict.items():
            color = (255, 0, 0)
            
            #線分が交わるか判定
            if hit_detection(wrist_pos, extended_pos, appliance_pos):
                if start_time_dict['Left'][appliance_name] == 0:
                    start_time_dict['Left'][appliance_name] = time.time()
                else: 
                    duration_time = time.time() - start_time_dict['Left'][appliance_name] 
                    # 一定時間以上交わったとき
                    if duration_time > L_DURATION:
                        cv2.putText(annotated_frame, f'L:{duration_time:.1f}', (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                        end = 'on'
                        duration_time = 0
                        ir_operation(appliance_name, end, 'Left')
                    else:
                        cv2.putText(annotated_frame, f'L:{duration_time:.1f}', (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)              
                color = (0, 0, 255)
                break
            else:
                start_time_dict['Left'][appliance_name] = 0

            
    cv2.line(annotated_frame, wrist_pos, extended_pos, color, 2)
    
# 右腕:チャンネル変更 
def r_ir_operation(annotated_frame, wrist_pos, appliance_dict, elbow_pos):
    global operation_time, pre_wrist_pos, pre_appliance_name, start_time_dict
    extended_pos = wrist_pos + 20 * (wrist_pos - elbow_pos)    #腕を伸ばした先
    # print(degree)
    if (time.time() - operation_time) > R_COOL_TIME or operation_time == 0:
        # print(degree < 45 or degree > 135)
        color = (255, 0, 0)
        for appliance_name, appliance_pos in appliance_dict.items():
            color = (255, 0, 0)
            # 線分が交わるか判定
            if hit_detection(wrist_pos, extended_pos, appliance_pos):
                if start_time_dict['Right'][appliance_name] == 0:
                    start_time_dict['Right'][appliance_name] = time.time()
                else:
                    duration_time = time.time() - start_time_dict['Right'][appliance_name]
                    
                    # 一定時間以上交わったときの手首の位置と操作対象を記録
                    if duration_time > R_DURATION:
                        pre_wrist_pos = wrist_pos
                        pre_appliance_name = appliance_name
                        cv2.putText(annotated_frame, f'R:{duration_time:.1f}', (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                        x, y, w, h = appliance_dict[appliance_name]
                        cv2.putText(annotated_frame, appliance_name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                        cv2.rectangle(annotated_frame, (x, y),(x+w, y+h), (255, 0, 0), 2)
                    else:
                        cv2.putText(annotated_frame, f'R:{duration_time:.1f}', (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                    
                color = (0, 0, 255)
                break
                    
            else:
                start_time_dict['Right'][appliance_name] = 0
                
        cv2.line(annotated_frame, wrist_pos, extended_pos, color, 2)

                
    # 線分が交わった状態から離れた場合
    #if isinstance(pre_appliance_name, str):
    # if pre_appliance_name is not None:
    #     appliance_pos = appliance_dict[pre_appliance_name]
        
    #     if not hit_detection(wrist_pos, ex_pos, appliance_pos):
    #         if wrist_pos[1] < pre_wrist_pos[1]:
    #             end = 'up'
    #         else:
    #             end = 'down'
            
    #         ir_operation(pre_appliance_name, end, 'Right')
            
    #         pre_appliance_name = None
            
  

def ir_operation(name, end, hand):
    global operation_time, operation_name
    operation_name = f'{name}-{end}'
    
    print(operation_name)
    irrp.ir_lightning(operation_name)

    operation_time = time.time()
    
    start_time_dict[hand][name] = 0
                        
    
# 操作対象と線の当たり判定
def hit_detection(pos1, pos2, appliance_pos):
    x, y, w, h = appliance_pos
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

    
    
