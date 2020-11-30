import socket
import cv2
import numpy as np
#from queue import Queue
from _thread import *
from multiprocessing import Queue
from labeling_module import LabelingModule

enclose_q = Queue()
lm = LabelingModule()
#socket에서 수신한 버퍼를 반환
def recvall(sock, count):
    # 바이트 문자열
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf
def send_threaded(Client_socket, addr, queue):
    print("Connected by : ", addr[0], " : ", addr[1])
    while True:
        try :
            data = Client_socket.recv(1024)
            if not data:
                print("Disconnected")
                break
            StringData = queue.get()
            Client_socket.send(str(len(StringData)).ljust(16).encode())
            Client_socket.send(StringData)
        except ConnectionResetError as e:
            print("Disconnected")
    Client_socket.close()
    
if __name__ == "__main__":
    lm.predict_process.start()

    RECV_HOST='192.168.35.227'
    RECV_PORT=9999 #RECV PORT
    
    #TCP 사용
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    print('Socket created')
    
    #CoreCloud IP, PortNumber set
    s.bind((RECV_HOST,RECV_PORT))
    print('Socket bind complete')
    # Edge Cloud 접속wait (클라이언트 연결을 10개까지 받음)
    s.listen(10)
    print('Socket now listening')
    
    #연결, conn 소켓 객체, addr socket binded addr
    conn,addr=s.accept()

    SEND_HOST = '192.168.35.87'
    SEND_PORT = 9998 #SEND PORT

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((SEND_HOST, SEND_PORT))
    server_socket.listen()
    while True:
        # client에서 받은 stringData length (==(str(len(stringData))).encode().ljust(16))
        length = recvall(conn, 16)
        stringData = recvall(conn, int(length))
        data = np.fromstring(stringData, dtype = 'uint8')
        #data decode
        cropped = cv2.imdecode(data, cv2.IMREAD_COLOR)
        cropped = cv2.resize(cropped, (48,48)) #Crop Image Resize
        result = lm.new_tensor(cropped) # Predict result
        lm.predict_process.join() # thread join
        edge_socket, addr = server_socket.accept()
        enclose_q.put(result)
        start_new_thread(send_threaded, (edge_socket, addr, enclose_q,))
        if(conn): #연결 끊어질 경우 loop 탈출
            break