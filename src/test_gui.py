import PySimpleGUI as sg
import utils


layout = [[sg.Text('カメラ起動しますか')],
          [sg.Button('起動', key='-YES-'), sg.Button('終了')]]

window = sg.Window('sample', layout)



event, values = window.read()

window.close()

# from playsound import playsound
# playsound("fanfare.mp3")
utils.test()

print(f'eventは{event}')