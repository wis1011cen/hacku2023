import PySimpleGUI as sg
import run
import ir.irrp as irrp
import time
import src.roi as roi
import json
import csv
import os
#flag = 0

#sg.theme('')
    

    #print(f.read())
    #reader = csv.reader(f)
   
    #for row in reader:
        #print(row)
     #       name = row[0]
            #x, y, w, h = map(int, row[1:])

            
layout1 = [[sg.Text('メインメニュー')],
          [sg.Button('カメラを起動', key='-camera-')],
          [sg.Button('赤外線信号を登録', key='-register-')],
          [sg.Button('終了', key='-exit-')]]
window1 = sg.Window('メインメニュー', layout1)


while True:

    event, values = window1.read()
    #print(values, event)
    #print(time.time())

#window.close()
    if event == '-camera-':
        #appliance_dict = dict()
        name_str = str()
        try:
            with open('src/roi.csv') as f:
                reader = csv.reader(f)
                for row in reader:
                    name_str += f'{row[0]}, '
        except:
            pass
               
        layout2 = [[sg.Checkbox('ミラー')],
                    [sg.Button('起動', key='-run-'), ],
                    [sg.Button('家電領域を選択', key='-select-')],
                    [sg.Text(f'保存済み一覧: {name_str}')],
                    [sg.Text('')],
                    [sg.Button('戻る', key='-back-')]]

        window2 = sg.Window('カメラモード', layout2)
        while True:
            
            event, values = window2.read()
            
            if event == '-run-':
                run.main(values[0])
            elif event == '-select-':
                roi.roi(values[0])
                window2.close()
            elif event == '-back-' or event == sg.WIN_CLOSED:
                window2.close()
                break
        
        
        
    elif event == '-register-':
        #filename = 'ir/codes'
#appliance_dict = dict()

        #print(key_list)
        #print(key for key in records.keys()

     
        while True:
            #key_list = list()
            key_str = str()
            try:
                with open('ir/codes', 'r') as f:
                    records = json.load(f)
                for key in records.keys():
                    key_str += f'{key}, '
            except:
                pass

               # key_list.append(key)
            #key_str = [key for key in records.keys()]
            layout3 = [[sg.Text('信号の名前'), sg.Input()],
                      [sg.Button('記録', key='-record-'), sg.Text('例: tv-on, tv-up, tv-down')],
                      [sg.Text('', key='-ACT-')],
                      [sg.Text('')],
                      [sg.Text('保存済み一覧')],
                      [sg.Text(key_str)],
                      [sg.Button('戻る', key='-back-')]]
            window3 = sg.Window('赤外線登録', layout3)
 
            event, values = window3.read()
            if event == '-record-' and len(values[0]) > 0:
                id = values[0]
        
                window3['-ACT-'].update(f"'{id}'のボタンを押してください")
                #event, values = window2.read()
                window3.Refresh()
                #time.sleep(3)
                value = irrp.ir_recording(id)
                if value == 'success':
                    window3.close()
                    sg.popup(f"'{id}'の信号を記録しました")
                elif value == 'short':
                    window3.close()
                    sg.popup("ERROR: コードが短すぎます。繰り返し押していませんか？")
                    
        
                    
            elif event == '-back-' or event == sg.WIN_CLOSED:
                window3.close()
                break
                



    elif event == sg.WIN_CLOSED or event == '-exit-':
        window1.close()
        break
    
    #else:
    

        

#flip_flag = values[0]
