import socket
PORT = 9999
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
server_socket.bind(('',PORT))

server_socket.listen(1)

client_socket,addr = server_socket.accept()

print('Connected by', addr)

while True:
        data = client_socket.recv(1024)
        if not data:
                pass
        print('Recieved from', addr, data.decode('utf-8'))
