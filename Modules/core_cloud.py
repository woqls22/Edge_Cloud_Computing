import socket
import cv2
import numpy as np
from multiprocessing import Queue

from labeling_module import LabelingModule
#socket에서 수신한 버퍼를 반환함.
def recvall(sock, count):
    # 바이트 문자열
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

    
if __name__ == "__main__":
    lm.predict_process.start()
    HOST='127.0.0.1'
    PORT=9999
    
    #TCP 사용
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    print('Socket created')
    
    #서버의 아이피와 포트번호 지정
    s.bind((HOST,PORT))
    print('Socket bind complete')
    # 클라이언트의 접속을 기다린다. (클라이언트 연결을 10개까지 받는다)
    s.listen(10)
    print('Socket now listening')
    
    #연결, conn에는 소켓 객체, addr은 소켓에 바인드 된 주소
    conn,addr=s.accept()
    while True:
        # client에서 받은 stringData의 크기 (==(str(len(stringData))).encode().ljust(16))
        length = recvall(conn, 16)
        stringData = recvall(conn, int(length))
        data = np.fromstring(stringData, dtype = 'uint8')
        
        #data를 디코딩한다.
        cropped = cv2.imdecode(data, cv2.IMREAD_COLOR)
        cropped = cv2.resize(cropped, (48,48)) #Crop Image Resize
        lm.new_tensor(cropped) # Predict result
        lm.predict_process.join() # thread join