import src.ssh_request as ssh_request

def main():
    client = ssh_request.connect()
    path = '~/project/gesture-control/ir'
    operation = input('Operation name:')
    print('Recording...')
    print(f"Press key for '{operation}'")
    
    stdin, stdout, stderr = client.exec_command(f'cd {path} && python3 irrp.py -r -g18 -f codes {operation} --no-confirm --post 130')
    stdout = list(stdout)
    print(stdout[2])
    
if __name__ == '__main__':
    main()