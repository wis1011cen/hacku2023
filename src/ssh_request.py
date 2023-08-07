import paramiko
import time
#import utils

#client = None
def connect():
    HOSTNAME = '192.168.1.144'
    USERNAME = 'pi'
    PASSWORD = 'password'
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print('SSH Connecting...')
    client.connect(hostname=HOSTNAME, username=USERNAME, password=PASSWORD)
    print('SSH Connected.')
    
    return client
    
def stream_on(client):
    #stdin, stdout, stderr = client.exec_command('pkill mjpg_streamer')
    path = '~/project/gesture-control/mjpg-streamer/mjpg-streamer-experimental'
    stdin, stdout, stderr = client.exec_command(f'cd {path} && ./mjpg_streamer -o "output_http.so -w ./www" -i "input_uvc.so -f 5 -q 10"')
    print('Streaming ON')


def stream_off(client):
    stdin, stdout, stderr = client.exec_command('pkill mjpg_streamer')
    print('Streaming OFF')
    
    
def ir_lighting(client, operation):
    print(operation)
    path = '~/project/gesture-control/ir '
    stdin, stdout, stderr = client.exec_command(f'cd {path} && python3 irrp.py -p -g17 -f codes {operation}')
    
    
    
    
def main():
    #global client
    #client = connect()
    with connect() as client:
    #stream_on()
    #time.sleep(10)
    #stream_off()
    #ir_lighting(client, 'tv')
    #ir_lighting(client, 'ac')
        ir_lighting(client, 'fan-down')
        
        
    #client.close()



        
if __name__ == '__main__':
    main()
